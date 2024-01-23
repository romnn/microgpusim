#![allow(clippy::missing_panics_doc)]

use quote::quote;
use syn::visit_mut::{self, VisitMut};

#[allow(dead_code)]
fn pretty_print<T>(input: T) -> String
where
    T: quote::ToTokens,
{
    let file: syn::File = syn::parse2(quote! {
        fn main() {
            #input
        }
    })
    .unwrap();
    prettyplease::unparse(&file)
}

#[derive(Debug)]
struct ControlFlowVisitorMut {
    current_reconvergence_point_id: usize,
    block: syn::Ident,
}

fn expr_diverges(expr: &syn::Expr) -> bool {
    matches!(
        expr,
        syn::Expr::If(syn::ExprIf { .. })
            | syn::Expr::ForLoop(syn::ExprForLoop { .. })
            | syn::Expr::While(syn::ExprWhile { .. })
            | syn::Expr::Loop(syn::ExprLoop { .. })
    )
}

impl VisitMut for ControlFlowVisitorMut {
    fn visit_block_mut(&mut self, node: &mut syn::Block) {
        let mut placeholders = Vec::new();
        for (i, stmt) in node.stmts.iter_mut().enumerate() {
            // placeholders.push((i, self.current_reconvergence_point_id));
            // visit_mut::visit_stmt_mut(self, stmt);
            match stmt {
                syn::Stmt::Macro(syn::StmtMacro { .. }) => {
                    // when encountering a macro, e.g.
                    // ```
                    // println!("{}", if 1 == 2 { true } else { false });
                    // ```
                    // we cannot parse and evaluate the token stream in the macro,
                    // so the best we can do is add a convergence point to be on
                    // the safer side.
                    //
                    // Note that nested control flow inside a macro expression can
                    // still cause issues.
                    placeholders.push((i, self.current_reconvergence_point_id));
                    self.current_reconvergence_point_id += 1;
                }
                syn::Stmt::Local(syn::Local {
                    init: Some(syn::LocalInit { expr, diverge, .. }),
                    ..
                }) if expr_diverges(expr) || diverge.is_some() => {
                    placeholders.push((i, self.current_reconvergence_point_id));
                    self.current_reconvergence_point_id += 1;
                }
                syn::Stmt::Expr(expr, _) if expr_diverges(expr) => {
                    let block = &self.block;
                    let reconvergence_point_id = self.current_reconvergence_point_id;
                    if let syn::Expr::If(syn::ExprIf { then_branch, .. }) = expr {
                        then_branch.stmts.insert(
                            0,
                            syn::parse2(quote! {
                                #block.took_branch(#reconvergence_point_id);
                            })
                            .unwrap(),
                        );
                    }
                    placeholders.push((i, reconvergence_point_id));
                    self.current_reconvergence_point_id += 1;
                }
                _ => {}
            }
            visit_mut::visit_stmt_mut(self, stmt);
        }

        placeholders.sort_by_key(|(i, _)| std::cmp::Reverse(*i));
        for (i, reconvergence_point_id) in placeholders {
            let block = &self.block;
            node.stmts.insert(
                i + 1,
                syn::parse2(quote! {
                    #block.reconverge_branch(#reconvergence_point_id);
                })
                .unwrap(),
            );

            node.stmts.insert(
                i,
                syn::parse2(quote! {
                    #block.start_branch(#reconvergence_point_id);
                })
                .unwrap(),
            );
        }
    }
}

#[proc_macro_attribute]
pub fn instrument_control_flow(
    _args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let mut func_ast: syn::ItemFn = syn::parse2(input.clone().into()).unwrap();

    let block: Option<&syn::Ident> = func_ast.sig.inputs.iter().find_map(|input| match input {
        syn::FnArg::Typed(syn::PatType { pat, .. }) => match pat.as_ref() {
            syn::Pat::Ident(syn::PatIdent { ident, .. }) => Some(ident),
            _ => None,
        },
        syn::FnArg::Receiver(_) => None,
    });
    let block = block.unwrap();

    // let mut test_ast: syn::Block = syn::parse2(quote! {{
    //     // let test = if false { 1 } else { 0 };
    //     let values = vec![];
    //     let a = "test";
    //     if a == "uwe" {
    //         println!("hi");
    //
    //         if a == "petra" {
    //             println!("ho");
    //         }
    //         // here
    //     }
    //     // here
    //     let _test = 0;
    //     let _test = 0;
    //     let _test = 0;
    //     loop {
    //         break;
    //     }
    //     // here
    //     let _test = 0;
    //     let _test = 0;
    //
    //     for i in 0..5 {
    //         println!(
    //             "i is {}",
    //             if i % 2 == 0 {
    //                 loop {
    //                     let _dead = 1;
    //                     break;
    //                 }
    //                 // here
    //                 "even"
    //             } else {
    //                 "odd"
    //             },
    //         );
    //         // here
    //     }
    //     let _test = 0;
    //     let _test = 0;
    //     // here
    //     let b = if true {
    //         let test = "";
    //         if a == a { }
    //         // here
    //         let access = values[0];
    //         "hi"
    //     } else { "ho" };
    //     // here
    // }})
    // .unwrap();

    // let mut test_ast: syn::Block = syn::parse2(quote! {{
    //     let a = "uwe";
    //     // start branch 1
    //     if a == "uwe" {
    //         // took branch 1
    //         let test = 3;
    //         // start branch 2
    //         if a == "petra" {
    //             // took branch 2
    //             let test = 3;
    //         }
    //         // end branch 2
    //     }
    //     // end branch 1
    // }})
    // .unwrap();

    // let out = &mut test_ast;
    let out = &mut func_ast.block;

    // add a reconvergence point to the beginning of the block
    out.stmts.insert(
        0,
        syn::parse2(quote! {
            #block.took_branch(0usize);
        })
        .unwrap(),
    );

    let mut visitor = ControlFlowVisitorMut {
        current_reconvergence_point_id: 1,
        block: block.clone(),
    };

    visitor.visit_block_mut(out);
    // println!("{}", pretty_print(out));
    // todo!();

    quote! {
        #func_ast
    }
    .into()
}

use proc_macro2::TokenStream;
use quote::quote;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};

// use quote::ToTokens;

#[allow(dead_code)]
fn pretty_print<T>(input: T) -> String
where
    T: quote::ToTokens,
{
    // let file = syn::parse_file(&input.to_string()).unwrap();
    let file: syn::File = syn::parse2(quote! {
        fn main() {
            #input
        }
    })
    .unwrap();
    prettyplease::unparse(&file)
}

struct ControlFlowVisitor;

impl<'ast> Visit<'ast> for ControlFlowVisitor {
    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        println!("found for loop");
        visit::visit_expr_for_loop(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        println!("found while loop");
        visit::visit_expr_while(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
        println!("found loop");
        visit::visit_expr_loop(self, node);
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        println!("found if: {:#?}", node);
        visit::visit_expr_if(self, node);
    }

    // fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
    //     println!("Function with name={}", node.sig.ident);
    //
    //     // Delegate to the default impl to visit any nested functions.
    //     visit::visit_item_fn(self, node);
    // }
}

#[derive(Debug)]
// struct ControlFlowVisitorMut<'ast> {
struct ControlFlowVisitorMut {
    current_reconvergence_point_id: usize,
    block: syn::Ident,
}

impl ControlFlowVisitorMut {
    fn expr_diverges(&self, expr: &syn::Expr) -> bool {
        match expr {
            syn::Expr::If(syn::ExprIf { .. })
            | syn::Expr::ForLoop(syn::ExprForLoop { .. })
            | syn::Expr::While(syn::ExprWhile { .. })
            | syn::Expr::Loop(syn::ExprLoop { .. }) => true,
            _ => false,
        }
    }
}

impl VisitMut for ControlFlowVisitorMut {
    fn visit_block_mut(&mut self, node: &mut syn::Block) {
        // node.stmts = node
        //     .stmts
        //     .clone()
        //     .into_iter()

        let mut placeholders = Vec::new();
        for (i, stmt) in node.stmts.iter_mut().enumerate() {
            // println!("=> statement {:#?}", stmt);
            visit_mut::visit_stmt_mut(self, stmt);
            match stmt {
                // syn::Stmt::Macro(syn::StmtMacro { mac, attrs, .. }) => {
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

                    // mac.parse_body()
                    // println!("=> macro attrs: {:#?}", attrs);
                    // println!("=> macro: {:#?}", mac);
                    // if self.expr_diverges(expr) || diverge.is_some() => {
                }
                syn::Stmt::Local(syn::Local {
                    init: Some(syn::LocalInit { expr, diverge, .. }),
                    ..
                }) if self.expr_diverges(expr) || diverge.is_some() => {
                    placeholders.push((i, self.current_reconvergence_point_id));
                    self.current_reconvergence_point_id += 1;
                    // println!("=> local expr: {:#?}", expr);
                    // println!("=> local diverge: {:#?}", diverge);
                }
                syn::Stmt::Expr(expr, _) if self.expr_diverges(expr) => {
                    placeholders.push((i, self.current_reconvergence_point_id));
                    self.current_reconvergence_point_id += 1;
                    // placeholders.push((i, syn::Stmt::Expr(syn::Expr::Block(()), _));
                }
                _ => {}
            }
        }

        placeholders.sort_by_key(|(i, _)| std::cmp::Reverse(*i));
        for (i, reconvergence_point_id) in placeholders {
            let block = &self.block;
            let stmt: syn::Stmt = syn::parse2(quote! {
                #block.reconverge(#reconvergence_point_id);
            })
            .unwrap();
            node.stmts.insert(i + 1, stmt);
        }
        // self.datas.insert(index, new_data);
        // visit_mut::visit_block_mut(self, node);
    }
}

#[proc_macro_attribute]
pub fn inject_reconvergence_points(
    args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let mut func_ast: syn::ItemFn = syn::parse2(input.clone().into()).unwrap();
    // let block = func_ast.block;
    // dbg!(&func_ast.attrs[0]);

    let block: Option<&syn::Ident> = func_ast.sig.inputs.iter().find_map(|input| match input {
        syn::FnArg::Typed(syn::PatType { pat, .. }) => match pat.as_ref() {
            syn::Pat::Ident(syn::PatIdent { ident, .. }) => Some(ident),
            _ => None,
        },
        _ => None,
    });
    let block = block.unwrap();
    // dbg!(&block);
    // let rec_point_id = 0;
    // let rec_point = quote! {
    //     #block.reconverge(#rec_point_id);
    // };
    // println!("{}", &rec_point);

    let mut test_ast: syn::Block = syn::parse2(quote! {{
        // let test = if false { 1 } else { 0 };
        let values = vec![];
        let a = "test";
        if a == "uwe" {
            println!("hi");

            if a == "petra" {
                println!("ho");
            }
            // here
        }
        // here
        let _test = 0;
        let _test = 0;
        let _test = 0;
        loop {
            break;
        }
        // here
        let _test = 0;
        let _test = 0;

        for i in 0..5 {
            println!(
                "i is {}",
                if i % 2 == 0 {
                    loop {
                        let _dead = 1;
                        break;
                    }
                    // here
                    "even"
                } else {
                    "odd"
                },
            );
            // here
        }
        let _test = 0;
        let _test = 0;
        // here
        let b = if true {
            let test = "";
            if a == a { }
            // here
            let access = values[0];
            "hi"
        } else { "ho" };
        // here
    }})
    .unwrap();

    // ControlFlowVisitor.visit_block(&func_ast.block);
    // ControlFlowVisitorMut.visit_block_mut(&mut func_ast.block);
    let mut visitor = ControlFlowVisitorMut {
        current_reconvergence_point_id: 0,
        block: block.clone(),
        // reconvergence_points: Vec::new(),
    };
    visitor.visit_block_mut(&mut test_ast);

    println!("{}", pretty_print(&test_ast));

    // dbg!(block.stmts.len());
    // dbg!(ast);
    todo!("todo");
    // input
    // let ast: syn::TokenStream = syn::parse2(input.into()).unwrap();
    // let (struct_name, generics, manifest_path) = parse_derive(&ast);
    // let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    // let x = format!(
    //     r#"
    //     fn dummy() {{
    //         println!("entering");
    //         println!("args tokens: {{}}", {args});
    //         println!("input tokens: {{}}", {input});
    //         println!("exiting");
    //     }}
    // "#,
    //     args = args.into_iter().count(),
    //     input = input.into_iter().count(),
    // );
    //
    // x.parse().expect("Generated invalid tokens")
}

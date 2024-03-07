#![allow(dead_code)]

#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate pest_ast;
#[macro_use]
extern crate pest;

mod ast;
mod ptx;

use crate::ptx::Rule;
use ast::{ASTNode, FunctionDeclHeader, ParseError};
use color_eyre::eyre;
use pest::iterators::Pair;
use pest::Parser;
use std::fs;
use std::path::{Path, PathBuf};

fn walk(pair: Pair<Rule>) -> eyre::Result<ASTNode> {
    match pair.as_rule() {
        Rule::function_defn => {
            let inner = pair.into_inner().map(|p| walk(p));
            println!("{:?}", inner.collect::<eyre::Result<Vec<ASTNode>>>());
            // Ok(ASTNode::FunctionDefn { name: "test" })
            Ok(ASTNode::FunctionDefn {})
        }
        Rule::function_decl => {
            let inner = pair.into_inner().map(|p| walk(p));
            println!("{:?}", inner.collect::<eyre::Result<Vec<ASTNode>>>());
            Ok(ASTNode::FunctionDecl { name: "test" })
        }
        Rule::function_ident_param => {
            // extract identifier and param_list
            let inner = pair.into_inner().map(|p| walk(p));
            println!("{:?}", inner.collect::<eyre::Result<Vec<ASTNode>>>());
            Ok(ASTNode::EOI)
        }
        Rule::function_decl_header => {
            let header = match pair.into_inner().next().map(|p| p.as_rule()) {
                Some(Rule::function_decl_header_entry) => Ok(FunctionDeclHeader::Entry),
                Some(Rule::function_decl_header_visible_entry) => {
                    Ok(FunctionDeclHeader::VisibleEntry)
                }
                Some(Rule::function_decl_header_weak_entry) => Ok(FunctionDeclHeader::WeakEntry),
                Some(Rule::function_decl_header_func) => Ok(FunctionDeclHeader::Func),
                Some(Rule::function_decl_header_visible_func) => {
                    Ok(FunctionDeclHeader::VisibleFunc)
                }
                Some(Rule::function_decl_header_weak_func) => Ok(FunctionDeclHeader::WeakFunc),
                Some(Rule::function_decl_header_extern_func) => Ok(FunctionDeclHeader::ExternFunc),
                _ => Err(ParseError::Unexpected(
                    "expected valid function decl header",
                )),
            }?;
            Ok(ASTNode::FunctionDeclHeader(header))
        }
        Rule::statement_block => {
            let inner = pair.into_inner().map(|p| walk(p));
            println!("{:?}", inner.collect::<eyre::Result<Vec<ASTNode>>>());
            Ok(ASTNode::EOI)
        }
        Rule::version_directive => {
            let mut iter = pair.into_inner();
            let double = iter.next().map(|p| walk(p)).unwrap();
            let newer = iter.next().map(|v| v.as_str() == "+").unwrap_or(false);

            match double {
                Ok(ASTNode::Double(version)) => Ok(ASTNode::VersionDirective { version, newer }),
                _ => unreachable!(),
            }
        }
        Rule::target_directive => {
            let identifiers: Vec<&str> = pair
                .into_inner()
                .flat_map(|id| match id.as_rule() {
                    Rule::identifier => Some(id.as_str()),
                    _ => None,
                })
                .collect();
            Ok(ASTNode::TargetDirective(identifiers))
        }
        Rule::address_size_directive => {
            let size: u32 = pair
                .into_inner()
                .next()
                .and_then(|s| s.as_str().parse().ok())
                .unwrap();
            Ok(ASTNode::AddressSizeDirective(size))
        }
        Rule::file_directive => {
            let mut inner = pair.into_inner().map(|p| walk(p));
            let id: usize = match inner.next() {
                Some(Ok(ASTNode::SignedInt(value))) => Ok(value.try_into()?),
                Some(Ok(ASTNode::UnsignedInt(value))) => Ok(value.try_into()?),
                _ => Err(ParseError::Unexpected("expected id")),
            }?;
            let path: PathBuf = match inner.next() {
                Some(Ok(ASTNode::Str(value))) => Ok(value.into()),
                _ => Err(ParseError::Unexpected("expected file path")),
            }?;
            let size: Option<usize> = match inner.next() {
                Some(Ok(ASTNode::SignedInt(value))) => Some(value.try_into()?),
                Some(Ok(ASTNode::UnsignedInt(value))) => Some(value.try_into()?),
                _ => None,
            };
            let lines: Option<usize> = match inner.next() {
                Some(Ok(ASTNode::SignedInt(value))) => Some(value.try_into()?),
                Some(Ok(ASTNode::UnsignedInt(value))) => Some(value.try_into()?),
                _ => None,
            };
            Ok(ASTNode::FileDirective {
                id,
                path,
                size,
                lines,
            })
        }
        Rule::identifier => Ok(ASTNode::Identifier(pair.as_str())),
        Rule::string => Ok(ASTNode::Str(pair.as_str())),
        Rule::double => {
            // let value = pair.as_str();
            // todo
            Ok(ASTNode::Double(0f64))
        }
        Rule::integer => {
            let value = pair.as_str();
            let unsigned = value.ends_with("U");
            if value.starts_with("0b") || value.starts_with("0B") {
                // binary
                return if unsigned {
                    Ok(ASTNode::UnsignedInt(u64::from_str_radix(
                        &value[2..value.len() - 1],
                        2,
                    )?))
                } else {
                    Ok(ASTNode::SignedInt(i64::from_str_radix(&value[2..], 2)?))
                };
            }
            if value.ends_with("U") {
                Ok(ASTNode::UnsignedInt(
                    value[..value.len() - 1].parse::<u64>()?,
                ))
            } else {
                Ok(ASTNode::SignedInt(value.parse::<i64>()?))
            }
            // let decimal = ;
            // hex: sscanf(yytext,"%x", &yylval->int_value
            // decimal: atoi(yytext)
        }
        Rule::EOI => Ok(ASTNode::EOI),
        other => {
            eprintln!("unhandled rule: {:?}", other);
            Ok(ASTNode::EOI)
        } // Rule::number => str::parse(pair.as_str()).unwrap(),
          // Rule::sum => {
          //     let mut pairs = pair.into_inner();

          //     let num1 = pairs.next().unwrap();
          //     let num2 = pairs.next().unwrap();

          //     process(num1) + process(num2)
          // }
    }
}

pub fn gpgpu_ptx_sim_load_ptx_from_filename(path: &Path) -> eyre::Result<u32> {
    let source = fs::read_to_string(path)?;
    // let source = String::from_utf8(fs::read(path)?)?;
    let parse_tree = ptx::Parser::parse(ptx::Rule::program, &source)?;

    // let ast: Program = parse_tree.try_into()?;
    // Program::from(&parse_tree);
    let ast = parse_tree
        // .iter()
        // .flat_map(|pair| walk(pair))
        .map(|pair| walk(pair))
        // match pair.as_rule() {
        // Rule::version_directive => {
        //   println!("{:?}", pair);
        //   // let mut pairs = pair.into_inner();
        //   let version = 0.1f64;
        //   let newer = false;
        //   // let version = pairs.next().unwrap();
        //   // let newer = pairs.next().ok();
        //   Some(Statement::Directive(Directive::Version { version, newer }))
        // }
        // Rule::EOI => None,
        // other => {
        //   eprintln!("unhandled rule: {:?}", other);
        //   None
        // }
        // )
        .collect::<eyre::Result<Vec<ASTNode>>>()?;
    println!("ast = {:#?}", ast);

    // for record in parse_tree {
    // println!("{:?}", record.as_rule());
    // match record.as_rule() {
    //     Rule::directive => {
    //         record_count += 1;

    //         for field in record.into_inner() {
    //             field_sum += field.as_str().parse::<f64>().unwrap();
    //         }
    //     }
    //     Rule::EOI => (),
    //     other => panic!("unhandled rule: {}", other),
    // }
    // }

    // println!("parse tree = {:#?}", parse_tree);
    // let ast: Program = File::from_pest(&mut parse_tree).expect("infallible");
    // println!("syntax tree = {:#?}", syntax_tree);
    // println!();
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use color_eyre::eyre;
    use std::path::PathBuf;

    macro_rules! ast_tests {
    ($($name:ident: $value:expr,)*) => {
    $(
        #[test]
        fn $name() -> eyre::Result<()> {
            let (rule, source, expected) = $value;
            let nodes = ptx::Parser::parse(rule, &source)?
                .map(|p| walk(p))
                .collect::<eyre::Result<Vec<ASTNode>>>()?;
            assert_eq!(Some(expected), nodes.into_iter().next());
            Ok(())
        }
    )*
    }
}

    ast_tests! {
        ast_integer_decimal_0: (ptx::Rule::integer, "0", ASTNode::SignedInt(0)),
        ast_integer_decimal_1: (ptx::Rule::integer, "-12", ASTNode::SignedInt(-12)),
        ast_integer_decimal_2: (ptx::Rule::integer, "12U", ASTNode::UnsignedInt(12)),
        ast_integer_decimal_3: (ptx::Rule::integer, "01110011001", ASTNode::SignedInt(1110011001)),
        ast_integer_binary_0: (ptx::Rule::integer, "0b01110011001", ASTNode::SignedInt(921)),
        ast_integer_binary_1: (ptx::Rule::integer, "0b01110011001U", ASTNode::UnsignedInt(921)),
    }

    #[cfg(feature = "local-data")]
    #[test]
    fn build_ast() -> eyre::Result<()> {
        let ptx_file = PathBuf::from("../kernels/mm/small.ptx");
        gpgpu_ptx_sim_load_ptx_from_filename(&ptx_file)?;
        Ok(())
    }

    //     kernelData * getKernelFunctionHelper(const char * bytes, unsigned int size, char * kernel, int kernelNumber) {
    // 	char* target = 0;
    // 	int index = 0;
    // 	kernelData* kern = 0;
    // 	unsigned long long maxLoc = size;
    //
    // 	if(kernel) {
    // 		target = (char*) malloc(strlen(kernel)+1);
    // 		strcpy(target, kernel);
    // 	}
    //
    // 	//Search each ELF for target kernel:
    // 	unsigned long long loc = 0x0;
    // 	while(!kern && loc < size) {
    // 		//Deal with padding and etc.
    // 		if(loc % 4 != 0) {
    // 			loc += 4 - (loc % 4);
    // 		}
    // 		while(true) {
    // 			if(loc >= size || *((unsigned int*)(bytes + loc)) == 0xba55ed50) {
    // 				break;
    // 			} else {
    // 				loc += 4;
    // 			}
    // 		}
    // 		if(loc >= size) {
    // 			break;
    // 		}
    //
    // 		fatHeader * head = (fatHeader*) &bytes[loc];
    // 		int headloc = loc;
    //
    // 		//Check magic number:
    // 		if(head->magic != 0xba55ed50 || head->unknown != 0x00100001) {
    // 			if(!loc) {
    // 				fprintf(stderr, "WARNING: unrecognized magic number for CUDA fatbin\n");
    // 			}
    // 			else {
    // 				fprintf(stderr, "SANITY CHECK ERROR ~1052: no magic number; possible misaligned fatbin.\n");
    // 			}
    // 		}
    //
    // 		loc += 0x10;
    // 		if(loc >= maxLoc) {
    // 			fprintf(stderr, "SANITY CHECK ERROR ~1044: fatbin is located out of bounds.\n");
    // 			return 0;
    // 		}
    // 		while(!kern && loc < head->size + headloc + 0x10) {
    // 			unsigned int * type = (unsigned int*) &bytes[loc];
    // 			unsigned int * offset = (unsigned int*) &bytes[loc + 4];
    // 			unsigned long long * size = (unsigned long long*) &bytes[loc + 8];
    //
    // 			int architecture = bytes[loc + 28];
    //
    // 			if(*size > headloc + head->size || loc + *offset + *size > headloc + 0x10 + head->size) {
    // 				fprintf(stderr, "SANITY CHECK FAILED ~1053: fatbin values out of bounds.\n");
    // 				return 0;
    // 			}
    //
    // 			loc += *offset;
    // 			if(loc >= maxLoc) {
    // 				fprintf(stderr, "SANITY CHECK FAILED ~1060: fatbin loc out of bounds.\n");
    // 				return 0;
    // 			}
    //
    // 			if((*type & 0xffff) == 0x2) {//this part of the fatbin contains an ELF
    // 				//Parse raw ELF data:
    // 				ELF * elf = bytes2ELF(bytes + loc);
    // 				loc += *size;
    // 				if(!elf) {
    // 					cerr << "SANITY CHECK ERROR em~885: unable to parse ELF.\n";
    // 					return 0;
    // 				}
    // 				if(loc > maxLoc) {
    // 					fprintf(stderr, "SANITY CHECK ERROR em~889: fatbin loc out of bounds.\n");
    // 					return 0;
    // 				}
    //
    // 				//Look for kernel code section:
    // 				int scnIndex = 0;
    // 				int numSections = getNumSections(elf);
    // 				for(int x = 0; x < numSections; x++) {
    // 					ELF_Section section = getSection(elf, x);
    // 					const ELF_SHeader shdr = getHeader(elf, section);
    // 					const char * name = getName(elf, section);
    // 					scnIndex = x;
    //
    // 					if(!target) {
    // 						bool containsKernel = !strncmp(name, ".text.", 6);
    // 						if(containsKernel) {
    // 							if(index == kernelNumber) {
    // 								target = (char*) malloc(strlen(name) + 2); //note: the extra malloc'd space avoids string errors elsewhere
    // 								strcpy(target, name + 6);
    // 							}
    // 							else {
    // 								index++;
    // 							}
    // 						}
    // 					}
    //
    // 					if(target && strlen(name) > 6 && !strcmp(name + 6, target)) {//this section contains the kernel function
    // 						//Copy data into a single char array:
    // 						char * bytes = (char*) malloc(shdr.sh_size);
    // 						memcpy(bytes, getSectionData(elf, section), shdr.sh_size);
    //
    // 						//Prepare return value:
    // 						kern = (kernelData*) malloc(sizeof(kernelData));
    // 						kern->sharedMemory = 0;
    // 						kern->min_stack_size = -1;
    // 						kern->max_stack_size = -1;
    // 						kern->frame_size = -1;
    // 						kern->bytes = bytes;
    // 						kern->name = target;
    // 						kern->arch = architecture;
    // 						kern->functionNames = 0;
    // 						kern->numBytes = shdr.sh_size;
    // 						kern->numRegisters = shdr.sh_info >> 24;
    // 						kern->symIndex = shdr.sh_info & 0xff;
    //
    // 						break;
    // 					}
    // 				}
    //
    // 				if(kern) {
    // 					for(int x = 0; x < numSections; x++) {
    // 						ELF_Section section = getSection(elf, x);
    // 						const ELF_SHeader shdr = getHeader(elf, section);
    // 						const char * name = getName(elf, section);
    //
    // 						//If section contains shared memory data, note size of shared memory
    // 						if(!strncmp(".nv.shared.", name, 11) && !strcmp(name + 11, kern->name)) {
    // 							kern->sharedMemory = shdr.sh_size;
    // 						}
    //
    // 						//Elseif symbol table, get subroutine names (if we can)
    // 						else if(shdr.sh_type == SHT_SYMTAB) {
    // 							if(shdr.sh_size % shdr.sh_entsize) {
    // 								cerr << "SANITY CHECK ERROR em~956: fractional symbol count.\n";
    // 							}
    //
    // 							//Find & change appropriate values in this symbol table:
    // 							int numSyms = shdr.sh_size / shdr.sh_entsize;
    // 							for(int y = 0; y < numSyms; y++) {
    // 								const ELF_Sym sym = getSymbol(elf, section, y);
    // 								if(sym.st_info == 0x22 && sym.st_shndx == scnIndex) {
    // 									const char * symName = getName(elf, shdr, sym);
    // 									char * copy = (char*) malloc(strlen(symName) + 1);
    // 									strcpy(copy, symName);
    // 									addLast(&kern->functionNames, copy);
    // 								}
    // 							}
    // 						}
    //
    // 						//Elseif .nv.info section, get local memory metadata
    // 						else if(!strcmp(".nv.info", name)) {
    // 							const char * bytes = getSectionData(elf, section);
    //
    // 							//Find appropriate values in section data:
    // 							for(unsigned int x = 0; x < shdr.sh_size;) {
    // 								CUDA_INFO * ci = (CUDA_INFO*)(bytes+x);
    //
    // 								if(ci->format > maxFormat || ci->format < minFormat) {
    // 									cerr << "ERROR: EIFMT type (0x" << std::hex << (int)ci->format << std::dec << ") out of range.\n";
    // 								}
    // 								if(ci->attribute > maxAttribute || ci->attribute < minAttribute) {
    // 									//cerr << "ERROR: EIATTR type (0x" << std::hex << (int)ci->attribute << std::dec << ") out of range.\n";
    // 								}
    //
    // 								int datasize = 0;
    // 								if(ci->format == EIFMT_NVAL) {
    // 									datasize = 0;
    // 								} else if(ci->format == EIFMT_BVAL) {
    // 									datasize = 1;
    // 								} else if(ci->format == EIFMT_HVAL) {
    // 									datasize = 2;
    // 								} else if(ci->format == EIFMT_SVAL) {
    // 									//TODO I don't know if this is correct for all attribute types:
    // 									datasize = 2;
    // 									short * sdata = (short*)ci->data;
    // 									datasize += sdata[0];
    // 								}
    //
    // 								if(ci->attribute == EIATTR_MIN_STACK_SIZE) {
    // 									if(ci->format == EIFMT_SVAL) {
    // 										if(datasize == 10) {
    // 											int * temp = (int*) (ci->data + 2);
    // 											int funcid = temp[0];
    // 											if(funcid == kern->symIndex) {
    // 												kern->min_stack_size = temp[1];
    // 											}
    // 										} else {
    // 											cerr << "ERROR: Unexpected datasize (" << datasize << ") for min_stack_size.\n";
    // 										}
    // 									} else {
    // 										cerr << "ERROR: Unexpected format for min_stack_size.\n";
    // 									}
    // 								}
    // 								if(ci->attribute == EIATTR_MAX_STACK_SIZE) {
    // 									if(ci->format == EIFMT_SVAL) {
    // 										if(datasize == 10) {
    // 											int * temp = (int*) (ci->data + 2);
    // 											int funcid = temp[0];
    // 											if(funcid == kern->symIndex) {
    // 												kern->max_stack_size = temp[1];
    // 											}
    // 										} else {
    // 											cerr << "ERROR: Unexpected datasize (" << datasize << ") for max_stack_size.\n";
    // 										}
    // 									} else {
    // 										cerr << "ERROR: Unexpected format for max_stack_size.\n";
    // 									}
    // 								} else if(ci->attribute == EIATTR_FRAME_SIZE) {
    // 									if(ci->format == EIFMT_SVAL) {
    // 										if(datasize == 10) {
    // 											int * temp = (int*) (ci->data + 2);
    // 											int funcid = temp[0];
    // 											if(funcid == kern->symIndex) {
    // 												kern->frame_size = temp[1];
    // 											}
    // 										} else {
    // 											cerr << "ERROR: Unexpected datasize (" << datasize << ") for frame_size.\n";
    // 										}
    // 									} else {
    // 										cerr << "ERROR: Unexpected format for frame_size.\n";
    // 									}
    // 								}
    //
    // 								x += datasize + 2;
    // 							}
    // 						}
    // 					}
    // 				}
    //
    // 				cleanELF(elf);
    // 			}
    // 			else {//not an ELF, ignore it
    // 				loc += *size;
    // 				if(loc > maxLoc) {
    // 					fprintf(stderr, "SANITY CHECK FAILED em~1058: fatbin loc out of bounds.\n");
    // 					return 0;
    // 				}
    // 			}
    // 		}
    // 	}
    //
    // 	//Cleanup & return:
    // 	if(kern && kern->bytes) {
    // 		return kern;
    // 	}
    // 	else {
    // 		if(target) {
    // 			free(target);
    // 		}
    // 		return 0;
    // 	}
    // }
    //

    #[repr(usize)]
    #[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub enum RegionKind {
        PTX = 1,
        ELF = 2,
    }

    #[cfg(feature = "local-data")]
    #[test]
    /// see https://pdfs.semanticscholar.org/5096/25785304410039297b741ad2007e7ce0636b.pdf
    /// see https://dl.acm.org/doi/abs/10.5555/3314872.3314900
    /// see https://github.com/daadaada/turingas/blob/master/turingas/cubin.py
    fn read_ptx_section() -> eyre::Result<()> {
        // use object::{FileHeader, Object, ObjectSection};
        use object::read::elf::FileHeader;
        use object::read::elf::{SectionHeader, SectionTable};
        // use object::read::Object;
        // use object::read::elf::{ElfSection64, FileHeader, Rela, SectionHeader, SectionTable};
        // use object::read::ObjectSection;
        use object::Endianness;
        // use object::ReadRef;
        // use object::{elf, ReadRef, SectionIndex};

        let binary = fs::read("/home/roman/dev/box/test-apps/vectoradd/vectoradd_l1_disabled")?;
        let kind = object::FileKind::parse(&*binary)?;
        assert_eq!(kind, object::FileKind::Elf64);

        let elf = object::elf::FileHeader64::<Endianness>::parse(&*binary)?;
        let endianness = elf.endian()?;
        dbg!(&endianness);
        let e_machine = elf.e_machine(endianness);
        dbg!(&e_machine);
        // elf::FileHeader64 < Endianness >> (data);
        //
        // let out_data = match kind {
        //     object::FileKind::Elf32 => copy_file::<elf::FileHeader32<Endianness>>(in_data).unwrap(),
        //     object::FileKind::Elf64 => copy_file::<elf::FileHeader64<Endianness>>(in_data).unwrap(),
        //     _ => {
        //         eprintln!("Not an ELF file");
        //         process::exit(1);
        //     }
        // };
        //
        // let file = object::File::parse(&*binary_data)?;
        // let sections: SectionTable<'_, _> = file.sections(endianness, &*binary)?;

        //Locate fatbin section on CUDA 4.0
        // https://github.com/decodecudabinary/Decoding-CUDA-Binary/blob/master/tools/src/elfmanip.cpp#L1026
        // target = ".rodata";
        // for(int x = 0; x < numSections; x++) {
        //     ELF_Section section = getSection(elf, x);
        //     const ELF_SHeader shdr = getHeader(elf, section);
        //     const char * name = getName(elf, section);
        //
        //     //Check if section is .nv_fatbin:
        //     if(name && !strcmp(target, name)) {
        //         const char * bytes = getSectionData(elf, section);
        //         kernelData* answer = getKernelFunctionHelper(bytes, shdr.sh_size, kernel, kernelNumber);
        //         cleanELF(elf);
        //         return answer;
        //     }
        // }

        let sections: SectionTable<_> = elf.sections(endianness, &*binary)?;
        for section in sections.iter() {
            // let string_table = section.strings(endianness, &*binary)?.unwrap_or_default();
            // dbg!(string_table);
            // let name = section.sh_name(endianness);
            let section_name = sections.section_name(endianness, section)?;
            // let name = section.name(endianness, string_table)?;
            // let Ok(name) = section.name(endianness, string_table) else {
            //     continue;
            // };
            // let section_name = String::from_utf8_lossy(section_name)?;
            let section_name = std::str::from_utf8(section_name)?;
            println!("{}", section_name);
            match section_name {
                ".nv_fatbin" | ".nvFatBinSegment" => {} // | ".text" => {}
                _ => continue,
            };
            // ".text.func" is the kernel name?
            // e_machine
            // e_machine(&self, endian: Self::Endian) -> u16
            println!("========");
            // the first 8 bytes are the .nv_fatbin magic number,
            // and the remaining eight bytes contain the size of the
            // rest of the region.
            // The rest of the region alternates between detailed
            // headers and the embedded file (ELF, PTX, or cubin) which
            // the detailed header describes.
            // In the detailed header, the first 4-byte word contains the
            // embedded file’s type and ptxas flags; the lower two bytes
            // have a value of 2 for GPU ELF files.
            // The second word is the offset of the embedded file,
            // relative to the start of this detailed header.
            // The dword comprising the third and fourth words holds
            // the size of the embedded file.
            // The seventh word is the code version,
            // which is dependent on the compiler.
            // The eighth word contains the target architecture - a value
            // of 20 for compute capability 2.0, a value of 35 for compute
            // capability 3.5, etcetera.
            //
            // The rest of the detailed header contains less important
            // metadata, such as the operating system or the source code’s
            // filename.
            // Another section of the CPU ELF that is unique to CUDA
            // programs is called .nvFatBinSegment.
            // It contains metadata about the .nv_fatbin section,
            // such as the starting addresses of its regions.
            // Its size is a multiple of six words (24 bytes),
            // where the third word in each group of six is an address
            // inside of the .nv_fatbin section.
            // If we modify the .nv_fatbin, then these addresses need to
            // be changed to match it.
            // let mut data = section.data(endianness, &*binary)?.to_owned();
            let data = section.data(endianness, &*binary)?;
            // let data = object::Bytes(data);
            // let data = object::Bytes(data);
            use bytes::{Buf, BufMut, Bytes, BytesMut};
            use object::ReadRef;
            // let mut buf = BytesMut::from(&mut *data);
            // let mut buf = Bytes::from(&*data);

            if section_name == ".nv_fatbin" {
                // let buf = Bytes::from(data).into_buf();

                // parse the large header
                // let mut nv_fatbin_magic_number = [0; 8];
                // buf.copy_to_slice(&mut nv_fatbin_magic_number);
                // let flag = data.read::<object::U32Bytes<_>>(4)?.get(endianness);
                let mut pos = 0;
                loop {
                    let nv_fatbin_magic_number = data
                        .read::<object::U64Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read nv fatbin magic number"))?
                        .get(endianness);
                    assert_eq!(nv_fatbin_magic_number, 0x00100001ba55ed50);

                    let region_size = data
                        .read::<object::U64Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read main header segment size"))?
                        .get(endianness);

                    // let nv_fatbin_magic_number = data.read_slice::<object::U32Bytes<_>>(8);

                    // .get(endianness);
                    // let mut segment_size = [0; 8];
                    // buf.copy_to_slice(&mut segment_size);
                    // println!("total size in bytes: {} {:x}", data.len(), data.len(),);
                    println!(
                        "\n\t => REGION SIZE: {:b} {:x}\n\n",
                        // u64::from_le_bytes(segment_size),
                        region_size,
                        region_size,
                    );
                    // println!(
                    //     "magic number: {:b} {:x}",
                    //     nv_fatbin_magic_number, nv_fatbin_magic_number,
                    // );
                    // assert_eq!(segment_size, 0x480);

                    // // let mut skip = [0; 4];
                    // // buf.copy_to_slice(&mut skip);
                    // // println!("skipped: {:?}", &skip);
                    //

                    let offset_start = pos;
                    let typ = data
                        .read::<object::U16Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read typ"))?
                        .get(endianness);

                    // assert_eq!(typ, 2); // 2 is ELF
                    let flags = data
                        .read::<object::U16Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read flags"))?
                        .get(endianness);

                    let offset = data
                        .read::<object::U32Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read offset"))?
                        .get(endianness);

                    let size = data
                        .read::<object::U64Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read size"))?
                        .get(endianness);

                    // skip 3 words
                    dbg!(String::from_utf8_lossy(
                        &data[pos as usize..pos as usize + 12]
                    ));
                    pos += 12;
                    // let _skipped = data
                    //     .read_bytes(12)
                    //     .map_err(|_| eyre::eyre!("failed to skip ahead"))?;

                    let code_version = data
                        .read::<object::U32Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read code version"))?
                        .get(endianness);

                    let architecture = data
                        .read::<object::U32Bytes<_>>(&mut pos)
                        .map_err(|_| eyre::eyre!("failed to read architecture"))?
                        .get(endianness);

                    dbg!(typ, flags, offset, size, code_version, architecture);

                    //this part of the fatbin contains an ELF
                    // ELF magic number: 0x464c457f

                    // let start = nt_size as usize;
                    // println!("STARTING FROM SEGMENT SIZE");
                    // dbg!(&start);
                    // for i in 0..8 {
                    //     println!("{:02X?}", &data[(start + (i * 32))..(start + (i + 1) * 32)]);
                    // }
                    // println!("STARTING FROM OFFSET");
                    // // let start = (offset - (pos - offset_start) as u32) as usize;
                    // let start = (offset_start + offset as u64) as usize;
                    // // let start = offset as usize;
                    // // let start = pos as usize;
                    // dbg!(&start);
                    // for i in 0..8 {
                    //     println!("{:02X?}", &data[(start + (i * 32))..(start + (i + 1) * 32)]);
                    // }

                    if typ == RegionKind::ELF as u16 {
                        let fatbin_elf_start = (offset_start + offset as u64) as usize;
                        let fatbin_elf_end = fatbin_elf_start + size as usize;

                        dbg!(fatbin_elf_start, fatbin_elf_end);
                        let fatbin_elf_data = &data[fatbin_elf_start..fatbin_elf_end];
                        let inner_kind = object::FileKind::parse(fatbin_elf_data)?;
                        assert_eq!(inner_kind, object::FileKind::Elf64);

                        let fatbin_elf =
                            object::elf::FileHeader64::<Endianness>::parse(fatbin_elf_data)?;
                        let fatbin_endianness = fatbin_elf.endian()?;
                        dbg!(&fatbin_endianness);
                        let fatbin_e_machine = fatbin_elf.e_machine(fatbin_endianness);
                        dbg!(&fatbin_e_machine);

                        assert_eq!(fatbin_e_machine, 190, "fatbin ELF is CUDA");

                        // iterate over kernel sections in ELF
                        let kernel_sections = fatbin_elf.sections(endianness, fatbin_elf_data)?;

                        // find the kernels in this region
                        use itertools::Itertools;
                        let kernel_names: Vec<_> = kernel_sections
                            .iter()
                            .filter_map(|sec| {
                                let name = kernel_sections.section_name(endianness, sec).ok()?;
                                let name = std::str::from_utf8(name).ok()?;
                                name.strip_prefix(".text.")
                                // if name.starts_with(".text.") {
                                //     // found a kernel code section
                                // } else {
                                //     None
                                // }
                            })
                            .dedup()
                            .sorted()
                            .collect();

                        dbg!(&kernel_names);

                        for kernel_name in &kernel_names {
                            // get shared memory data, note size of shared memory
                            let shmem = kernel_sections.section_by_name(
                                endianness,
                                format!(".nv.shared.{}", kernel_name).as_bytes(),
                            );
                            if let Some((_, shmem_section)) = shmem {
                                let shared_mem_bytes = shmem_section.sh_size(endianness);
                                dbg!(shared_mem_bytes);
                            }

                            // get shared memory data, note size of shared memory
                            let info = kernel_sections.section_by_name(
                                endianness,
                                format!(".nv.info.{}", kernel_name).as_bytes(),
                            );
                            if let Some((_, info_section)) = info {
                                let info_data = info_section.data(endianness, &*fatbin_elf_data);
                                // let shared_mem_bytes = info_section.sh_size(endianness);
                                // dbg!(shared_mem_bytes);
                            }
                        }
                        for (kernel_section_idx, kernel_section) in
                            kernel_sections.iter().enumerate()
                        {
                            // kernel_section.section
                            let kernel_section_name =
                                kernel_sections.section_name(endianness, kernel_section)?;
                            let kernel_section_name = std::str::from_utf8(kernel_section_name)?;
                            println!("=======================");
                            println!("{:>3} => {}", kernel_section_idx, kernel_section_name);

                            let kernel_section_data =
                                kernel_section.data(endianness, fatbin_elf_data)?;

                            // so turns out, the text sections are actually the
                            // SASS assembly instructions.
                            // This can be verified with usign cuobjdump -sass
                            println!("{}", String::from_utf8_lossy(kernel_section_data));
                            println!("{:02X?}", &kernel_section_data);
                            println!("=======================");

                            let section_type = kernel_section.sh_type(endianness);
                            let section_size = kernel_section.sh_size(endianness);
                            let section_entry_size = kernel_section.sh_entsize(endianness);
                            if section_type == object::elf::SHT_SYMTAB {
                                assert_eq!(
                                    section_size % section_entry_size,
                                    0,
                                    "fractional symbol count"
                                );
                            }
                        }
                        // let fatbin_data = section.data(endianness, &*binary)?;

                        pos = fatbin_elf_end as u64;
                    } else if typ == RegionKind::PTX as u16 {
                        let ptx_start = (offset_start + offset as u64) as usize;
                        let ptx_end = ptx_start + size as usize;

                        dbg!(ptx_start, ptx_end);
                        let ptx_data = &data[ptx_start..ptx_end];

                        println!("{}", String::from_utf8_lossy(&ptx_data));
                        println!("{:02X?}", &ptx_data);
                        println!("=======================");
                    }
                    // break;

                    // let typ = buf.get_u16_le();
                    // assert_eq!(typ, 2); // 2 is ELF
                    // let flags = buf.get_u16_le();
                    // let offset = buf.get_u32_le();
                    // let size = buf.get_u64_le(); // 4th
                    // 5,6,7
                    // let _ = buf.get_u32_le();
                    // let _ = buf.get_u32_le();
                    // let _ = buf.get_u32_le();
                    // let code_version = buf.get_u32_le();
                    // let architecture = buf.get_u32_le();
                    // let mut skip = [0; 20];
                    // buf.copy_to_slice(&mut skip);
                    // println!("skipped: {:?}", &skip);
                    // dbg!(architecture);
                }
            } else if section_name == ".nvFatBinSegment" {
                dbg!(data.len());
                let num_regions = data.len() / (6 * 4);
                dbg!(&num_regions);
                for region in 0..num_regions {
                    // read the third word in group of 6
                    let word = region * (6 * 4) + (3 * 4);
                    dbg!(&word);

                    let region_starting_address = data
                        .read_at::<object::U32Bytes<_>>(word as u64)
                        .map_err(|_| {
                            eyre::eyre!("failed to read starting address for region {}", region)
                        })?
                        .get(endianness);
                    dbg!(&region_starting_address);
                }
            }

            let data = String::from_utf8_lossy(&data);
            // println!("{}", data);
            println!("========");
            println!("\n\n\n");
            // println!("{:?}", section.data()?);
            // println!("{}", String::from_utf8_lossy(section.data()?));
        }
        Ok(())
    }
}

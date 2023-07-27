#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use playground::bindings;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::pin::Pin;

    fn is_mem_instruction(trace: &bindings::TraceEntry) -> bool {
        use bindings::op_type;
        use bindings::TraceInstrOpcode;
        // dbg!(&trace.op);
        // dbg!(&trace.opcode);
        if let op_type::EXIT_OPS | op_type::LOAD_OP | op_type::STORE_OP = trace.op {
            return true;
        }
        if trace.opcode == TraceInstrOpcode::OP_LDC {
            return true;
        }
        false
    }

    #[ignore = "todo"]
    #[test]
    fn test_parse_traces() -> eyre::Result<()> {
        use playground::trace_parser::TraceParser;

        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-100-32");
        let trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-1000-32");
        let trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-10000-32");
        let trace_dir = manifest_dir.join("results/simple_matrixmul/simple_matrixmul-32-32-32-32");
        let trace_dir = manifest_dir.join("results/simple_matrixmul/simple_matrixmul-32-32-64-32");
        let trace_dir =
            manifest_dir.join("results/simple_matrixmul/simple_matrixmul-64-128-128-32");

        let trace_parser = TraceParser::new(trace_dir.join("accelsim-trace/kernelslist.g"));
        let play_commands = trace_parser.parse_commandlist_file();

        let mut play_kernel_infos: HashMap<usize, _> = HashMap::new();
        for cmd in play_commands {
            if cmd.kind == playground::bindings::command_type::kernel_launch {
                let kernel_trace_info = trace_parser.parse_kernel_info(cmd.command);
                // kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
                //                                  &tconfig, tracer);
                play_kernel_infos.insert(
                    unsafe { kernel_trace_info.as_ref().unwrap() }.get_kernel_id() as usize,
                    kernel_trace_info,
                );
            }
        }

        let mut box_kernel_infos: HashMap<usize, _> = HashMap::new();
        let box_commands = crate::ported::parse_commands(trace_dir.join("trace/commands.json"))?;
        for cmd in box_commands {
            if let trace_model::Command::KernelLaunch(config) = cmd {
                box_kernel_infos.insert((config.id + 1) as usize, config);
            }
        }

        assert_eq!(
            play_kernel_infos.keys().collect::<Vec<_>>(),
            box_kernel_infos.keys().collect::<Vec<_>>()
        );
        for (kernel_id, play_kernel) in play_kernel_infos {
            let play_traces =
                unsafe { trace_parser.inner.get_next_threadblock_traces(play_kernel) };
            let mut play_traces: Vec<_> = play_traces.into_iter().collect();

            let box_kernel = &box_kernel_infos[&kernel_id];
            let box_traces =
                crate::ported::read_trace(trace_dir.join("trace").join(&box_kernel.trace_file))?;

            dbg!(play_traces.len());

            // filter play traces
            play_traces.retain(|trace| is_mem_instruction(trace));
            dbg!(play_traces.len());
            dbg!(box_traces.len());

            play_traces.sort_by_key(|trace| (trace.block_x, trace.block_y, trace.block_z));
            // play_traces.sort_by_key(|(a, b)| {
            //     let block_a = (a.block_x, a.block_y, a.block_z);
            //     let block_b = (b.block_x, b.block_y, b.block_z);
            //     ((block_a,)).cmp((block_b,))
            // });

            assert_eq!(box_traces.len(), play_traces.len());

            //     for trace in traces.iter() {
            //         dbg!(trace);
            //     }
            //
            //     // let traces: Vec<Vec<&playground::trace_parser::inst_trace_t>> = traces
            //     // let traces: Vec<Vec<&playground::warp_inst::warp_inst_t>> = traces
            //     //     .into_iter()
            //     //     .map(|t| t.inner.into_iter().collect())
            //     //     .collect();
            //     //
            //     // dbg!(traces.len());
            //     // for warp in &traces {
            //     //     dbg!(warp.len());
            //     //     for inst in warp {
            //     //         // let inst = playground::warp_inst::WarpInstr::new(
            //     //         // dbg!((&**inst as *const playground::warp_inst::warp_inst_t) as usize);
            //     //         // dbg!((*inst).empty());
            //     //         // dbg!((*inst).get_pc());
            //     //         // dbg!((*inst).warp_id());
            //     //         for w in 0..32 {
            //     //             // dbg!(inst.get_addr());
            //     //         }
            //     //     }
            //     // }
        }
        assert!(false, "all good");
        Ok(())
    }
}

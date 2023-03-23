#![allow(clippy::cast_possible_truncation)]

use anyhow::Result;
use rangemap::RangeMap;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use trace_model as model;

#[derive(Clone, Debug)]
pub struct MemoryAccesses<T, A>
where
    A: Clone + Eq,
{
    accesses: HashMap<(bool, Option<String>), Vec<T>>,
    allocations: RangeMap<u64, A>,
    bands: Vec<std::ops::Range<u64>>,
}

impl<T, A> Default for MemoryAccesses<T, A>
where
    A: Clone + Eq,
{
    fn default() -> Self {
        Self {
            accesses: HashMap::new(),
            allocations: RangeMap::new(),
            bands: Vec::new(),
        }
    }
}

impl MemoryAccesses<model::MemAccessTraceEntry, model::MemAllocation> {
    pub fn register_allocation(&mut self, alloc: model::MemAllocation) {
        let start = alloc.device_ptr;
        let end = alloc.device_ptr + alloc.bytes as u64;
        self.allocations.insert(start..end, alloc);
        self.bands.push(start..end);
    }

    pub fn add(&mut self, access: model::MemAccessTraceEntry, label: Option<String>) {
        let key = (access.instr_is_store, label);
        let accesses = self.accesses.entry(key).or_insert(vec![]);
        accesses.push(access);
    }

    /// Draw memory accesses to path
    ///
    /// # Errors
    /// When drawing or saving the result fails.
    pub fn draw(&mut self, path: impl AsRef<Path>) -> Result<()> {
        use plotters::prelude::*;

        // let size = (600, 400);
        // let path = path.as_ref().with_extension("svg");
        // let backend = SVGBackend::new(&path, size);

        let size = (2000, 1500);
        let path = path.as_ref().with_extension("png");
        let backend = BitMapBackend::new(&path, size);

        let root_area = backend.into_drawing_area();
        root_area.fill(&WHITE)?;

        let max_addr = self
            .accesses
            .values()
            .flatten()
            // .flat_map(|accesses| accesses)
            .flat_map(|access| access.addrs)
            .max()
            .unwrap_or(u64::MIN);
        let max_addr = max_addr.checked_add(32).unwrap_or(max_addr);

        let min_addr = self
            .accesses
            .values()
            .flatten()
            // .flat_map(|accesses| accesses)
            .flat_map(|access| access.addrs)
            .filter(|addr| *addr > 100)
            .min()
            .unwrap_or(0);
        let min_addr = min_addr.checked_sub(32).unwrap_or(min_addr);
        let max_time = self
            .accesses
            .values()
            // .flat_map(|accesses| accesses)
            .flatten()
            .map(|access| 32 * (access.warp_id + 1))
            .max()
            .unwrap_or_default();

        let font_size = 20;
        let font = ("monospace", font_size).into_font();
        let text_style = TextStyle::from(font).color(&BLACK);
        let x_range = 0..max_time;
        let y_range = min_addr..max_addr;

        dbg!(&x_range);
        dbg!(&y_range);

        let mut chart_ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 100)
            .set_label_area_size(LabelAreaPosition::Bottom, 2 * font_size)
            .caption("Memory accesses", text_style)
            .build_cartesian_2d(x_range.clone(), y_range)?;

        chart_ctx
            .configure_mesh()
            .y_label_formatter(&|y| format!("{y:#16x}"))
            .draw()?;

        for band in &self.bands {
            // The left upper and right lower corner of the rectangle
            let left_upper = (0, band.end);
            let right_lower = (x_range.end, band.start);
            let rect = Rectangle::new(
                [left_upper, right_lower],
                ShapeStyle {
                    color: BLUE.mix(0.5),
                    filled: true,
                    stroke_width: 3,
                },
            );
            chart_ctx.plotting_area().draw(&rect)?;
        }

        // [(t["warp_id"], 32*t["warp_id"]+tid, a) for tid, a in enumerate(t["addrs"]) if a > 0]
        // entries = sorted(entries, key=lambda x: x[0])
        // addresses = [e[2] for e in entries]
        // time = [e[1] for e in entries]

        for ((is_store, label), mut accesses) in &mut self.accesses {
            println!("drawing ({is_store}, {label:?})");

            accesses.sort_by(|a, b| a.warp_id.cmp(&b.warp_id));
            // dbg!(&accesses);

            let color = if *is_store { &RED } else { &BLUE };
            let series = accesses.iter_mut().flat_map(|access| {
                access
                    .addrs
                    .into_iter()
                    .enumerate()
                    .filter(|(_, addr)| *addr > 0)
                    .map(|(tid, addr)| {
                        let time = 32 * access.warp_id + tid as u32;
                        Circle::new((time, addr), 5, color)
                    })
            });
            chart_ctx
                .draw_series(series)?
                .label(if *is_store { "Store" } else { "Read" }.to_string())
                // .label(label.as_ref().cloned().unwrap_or("".to_string()))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));
        }
        chart_ctx
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()?;

        println!("finished drawing");
        Ok(())
    }
}

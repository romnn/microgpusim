use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

// type Point = (i32, i32);

#[derive(Clone, Debug)]
pub struct MemoryAccesses<T> {
    data: HashMap<(bool, Option<String>), Vec<T>>,
}

impl<T> Default for MemoryAccesses<T> {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl MemoryAccesses<super::MemAccessTraceEntry> {
    // pub fn access(&mut self, address: u64, store: bool, label: Option<String>) {
    pub fn add(&mut self, access: super::MemAccessTraceEntry, label: Option<String>) {
        let key = (access.instr_is_store, label);
        let accesses = self.data.entry(key).or_insert(vec![]);
        accesses.push(access);
    }

    pub fn draw(&mut self, path: impl AsRef<Path>) -> Result<()> {
        use plotters::prelude::*;

        let size = (2000, 1000);
        let size = (1000, 500);
        let size = (600, 400);
        let backend = SVGBackend::new(path.as_ref(), size);
        // let backend = BitMapBackend::new(path.as_ref(), size);
        let root_area = backend.into_drawing_area();
        root_area.fill(&WHITE)?;

        let min_addr = self
            .data
            .values()
            .flat_map(|accesses| accesses)
            .flat_map(|access| access.addrs)
            .filter(|addr| *addr > 0)
            .min()
            .unwrap_or(0);
        let max_addr = self
            .data
            .values()
            .flat_map(|accesses| accesses)
            .flat_map(|access| access.addrs)
            .max()
            .unwrap_or(u64::MIN);
        let max_time = self
            .data
            .values()
            .flat_map(|accesses| accesses)
            .map(|access| 32 * (access.warp_id + 1))
            .max()
            .unwrap_or_default();

        let font_size = 20;
        let font = ("monospace", font_size).into_font();
        let text_style = TextStyle::from(font).color(&BLACK);
        let x_range = 0..max_time;
        let y_range = min_addr - 32..max_addr + 32;
        dbg!(&x_range);
        dbg!(&y_range);
        let mut chart_ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 100)
            .set_label_area_size(LabelAreaPosition::Bottom, 2*font_size)
            .caption("Memory accesses", text_style)
            .build_cartesian_2d(x_range, y_range)?;

        chart_ctx
            .configure_mesh()
            .y_label_formatter(&|y| format!("{y:#16x}"))
            .draw()?;

        // [(t["warp_id"], 32*t["warp_id"]+tid, a) for tid, a in enumerate(t["addrs"]) if a > 0]
        // entries = sorted(entries, key=lambda x: x[0])
        // addresses = [e[2] for e in entries]
        // time = [e[1] for e in entries]

        for ((is_store, label), mut accesses) in self.data.iter_mut() {
            accesses.sort_by(|a, b| a.warp_id.cmp(&b.warp_id));
            // dbg!(&accesses);

            let color = if *is_store { &RED } else { &BLUE };
            let series = accesses.into_iter().flat_map(|access| {
                access
                    .addrs
                    .into_iter()
                    .enumerate()
                    .filter(|(_, addr)| *addr > 0)
                    .map(|(tid, addr)| {
                        let time = 32 * access.warp_id as u32 + tid as u32;
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
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()?;

        Ok(())
    }
}

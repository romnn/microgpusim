pub struct Style(indicatif::ProgressStyle);

impl Default for Style {
    fn default() -> Self {
        let style = indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap();
        let style = style.with_key(
            "eta",
            |state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
                let _ = write!(w, "{:.1}s", state.eta().as_secs_f64());
            },
        );
        let style = style.progress_chars("#-");
        Self(style)
    }
}

impl From<Style> for indicatif::ProgressStyle {
    fn from(val: Style) -> Self {
        val.0
    }
}

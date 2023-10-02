use playground_sys::cluster::cluster_bridge;

#[derive(Clone)]
pub struct Cluster<'a>(pub(crate) &'a cluster_bridge);

impl<'a> Cluster<'a> {
    // #[inline]
    #[must_use]
    pub fn core_sim_order(&self) -> Vec<usize> {
        self.0
            .get_core_sim_order()
            .into_iter()
            .map(|i| *i as usize)
            .collect()
    }
}

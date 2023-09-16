use petgraph::visit::{VisitMap, Visitable};
use petgraph::{algo, prelude::*};
use std::collections::HashSet;

pub trait UniqueGraph<N, E, Ix> {
    fn add_unique_edge(&mut self, a: NodeIndex<Ix>, b: NodeIndex<Ix>, weight: E) -> EdgeIndex<Ix>
    where
        E: PartialEq;

    fn find_node<W>(&self, weight: &W) -> Option<NodeIndex<Ix>>
    where
        W: PartialEq<N>;

    fn add_unique_node(&mut self, weight: N) -> NodeIndex<Ix>
    where
        N: PartialEq;
}

impl<N, E, D, Ix> UniqueGraph<N, E, Ix> for Graph<N, E, D, Ix>
where
    D: petgraph::EdgeType,
    Ix: petgraph::graph::IndexType,
{
    fn add_unique_edge(&mut self, a: NodeIndex<Ix>, b: NodeIndex<Ix>, weight: E) -> EdgeIndex<Ix>
    where
        E: PartialEq,
    {
        match self.find_edge(a, b) {
            Some(edge) if self.edge_weight(edge) == Some(&weight) => edge,
            _ => Graph::<N, E, D, Ix>::add_edge(self, a, b, weight),
        }
    }

    fn find_node<W>(&self, weight: &W) -> Option<NodeIndex<Ix>>
    where
        W: PartialEq<N>,
    {
        self.node_indices()
            .find(|idx| match self.node_weight(*idx) {
                Some(node) => *weight == *node,
                None => false,
            })
    }

    fn add_unique_node(&mut self, weight: N) -> NodeIndex<Ix>
    where
        N: PartialEq,
    {
        match self.find_node(&weight) {
            Some(idx) => idx,
            _ => Graph::<N, E, D, Ix>::add_node(self, weight),
        }
    }
}

pub fn all_simple_paths<TargetColl, G>(
    graph: G,
    from: G::NodeId,
    to: G::NodeId,
    // min_intermediate_nodes: usize,
    // max_intermediate_nodes: Option<usize>,
) -> impl Iterator<Item = TargetColl>
where
    G: petgraph::visit::NodeCount,
    G: petgraph::visit::IntoNeighborsDirected,
    G: petgraph::visit::IntoEdgesDirected,
    // <G as petgraph::visit::IntoNeighborsDirected>::NeighborsDirected: Neighbors<'a, E, Ix>,
    // G: petgraph::visit::Walker,

    // Neighbors<'a, E, Ix>
    // Ix: IndexType,
    // G: petgraph::graph::WalkNeighbors<G::NodeId>,
    G::NodeId: Eq + std::hash::Hash,
    G::EdgeId: Eq + std::hash::Hash,
    // TargetColl: FromIterator<G::NodeId>,
    TargetColl: FromIterator<(Option<G::EdgeId>, G::NodeId)>,
{
    use indexmap::IndexSet;
    // how many nodes are allowed in simple path up to target node
    // it is min/max allowed path length minus one, because it is more appropriate when implementing lookahead
    // than constantly add 1 to length of current path
    // let max_length = if let Some(l) = max_intermediate_nodes {
    //     l + 1
    // } else {
    //     graph.node_count() - 1
    // };
    //
    // let min_length = min_intermediate_nodes + 1;

    // list of visited nodes
    // let mut visited: IndexSet<G::NodeId> = IndexSet::from_iter(Some(from));
    let mut visited: IndexSet<(Option<G::EdgeId>, G::NodeId)> = IndexSet::from_iter([(None, from)]);
    // IndexSet::from_iter(Some(None, Some(from)));
    // list of childs of currently exploring path nodes,
    // last elem is list of childs of last visited node
    // let mut edges: petgraph::graph::WalkNeighbors<G::NodeId> =
    //     graph.neighbors_directed(from, Outgoing).detach();
    // let mut children = vec![];
    // for edge in graph.edges_directed(from, Outgoing) {
    //     dbg!(&edge.source());
    // }
    // while let Some(child) = edges.next(&graph) {
    //     children.push(child);
    // }
    // let mut stack = vec![children];
    // let mut stack = vec![edges];
    // let mut stack = vec![graph.neighbors_directed(from, Outgoing)];
    // struct EdgeNodeIter {
    //     inner: String
    // }
    //
    // impl Iterator for EdgeNodeIter {
    //     type Item = u32;
    //     fn next(&mut self) -> Option<Self::Item> {
    //         // Some(current)
    //     }
    // }
    // let initial_children = graph
    //     .edges_directed(from, Outgoing)
    //     .map(|edge| (edge.id(), edge.target()));
    let mut stack = vec![graph.edges_directed(from, Outgoing)];
    // .collect::<Vec<_>>()];

    std::iter::from_fn(move || {
        while let Some(children) = stack.last_mut() {
            if let Some(edge) = children.next() {
                // if let Some(child) = children.next() {
                let child = edge.target();
                if child == to {
                    let path = visited
                        .iter()
                        .cloned()
                        .chain([(Some(edge.id()), to)])
                        // ccchain(Some((Some(edge.id()), to)))
                        .collect::<TargetColl>();
                    return Some(path);
                } else if !visited.contains(&(Some(edge.id()), child)) {
                    visited.insert((Some(edge.id()), child));
                    // let new_children = graph
                    //     .edges_directed(from, Outgoing)
                    //     .map(|edge| (edge.id(), edge.target()));
                    stack.push(graph.edges_directed(child, Outgoing));
                    // stack.push(graph.neighbors_directed(child, Outgoing));
                }
            } else {
                stack.pop();
                visited.pop();
            }
        }
        None
    })
}

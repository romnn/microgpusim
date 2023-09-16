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
    G::NodeId: Eq + std::hash::Hash,
    TargetColl: FromIterator<G::NodeId>,
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
    let mut visited: IndexSet<G::NodeId> = IndexSet::from_iter(Some(from));
    // list of childs of currently exploring path nodes,
    // last elem is list of childs of last visited node
    let mut stack = vec![graph.neighbors_directed(from, Outgoing)];

    std::iter::from_fn(move || {
        while let Some(children) = stack.last_mut() {
            if let Some(child) = children.next() {
                // if visited.len() < max_length {
                if child == to {
                    // if visited.len() >= min_length {
                    let path = visited
                        .iter()
                        .cloned()
                        .chain(Some(to))
                        .collect::<TargetColl>();
                    return Some(path);
                    // }
                } else if !visited.contains(&child) {
                    visited.insert(child);
                    stack.push(graph.neighbors_directed(child, Outgoing));
                }
                // } else {
                //     if (child == to || children.any(|v| v == to)) && visited.len() >= min_length {
                //         let path = visited
                //             .iter()
                //             .cloned()
                //             .chain(Some(to))
                //             .collect::<TargetColl>();
                //         return Some(path);
                //     }
                //     stack.pop();
                //     visited.pop();
                // }
            } else {
                stack.pop();
                visited.pop();
            }
        }
        None
    })
}

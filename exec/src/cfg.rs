use super::model::MemInstruction;
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{algo, prelude::*};
use std::collections::HashSet;

#[derive(Debug)]
pub enum TraceNode {
    Branch {
        id: usize,
        instructions: Vec<MemInstruction>,
    },
    Reconverge {
        id: usize,
        instructions: Vec<MemInstruction>,
    },
}

impl std::fmt::Display for TraceNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Branch { id, instructions } => {
                write!(f, "Branch(id={id}, inst={})", instructions.len())
            }
            Self::Reconverge { id, instructions } => {
                write!(f, "Reconverge(id={id}, inst={})", instructions.len())
            }
        }
    }
}

impl TraceNode {
    #[inline]
    pub fn id(&self) -> usize {
        match self {
            Self::Branch { id, .. } | Self::Reconverge { id, .. } => *id,
        }
    }

    #[inline]
    pub fn instructions(&self) -> &[MemInstruction] {
        match self {
            Self::Branch { instructions, .. } | Self::Reconverge { instructions, .. } => {
                instructions
            }
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Node {
    Branch(usize),
    Reconverge(usize),
}

impl Node {
    #[inline]
    pub fn id(&self) -> usize {
        match self {
            Self::Branch(id) | Self::Reconverge(id) => *id,
        }
    }
}

impl PartialEq<TraceNode> for Node {
    fn eq(&self, other: &TraceNode) -> bool {
        match (self, other) {
            (Self::Branch(branch_id), TraceNode::Branch { id, .. }) => branch_id == id,
            (Self::Reconverge(branch_id), TraceNode::Reconverge { id, .. }) => branch_id == id,
            _ => false,
        }
    }
}

pub type CFG = petgraph::graph::DiGraph<Node, bool>;
pub type ThreadCFG = petgraph::graph::DiGraph<TraceNode, bool>;

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
) -> impl Iterator<Item = TargetColl>
where
    G: petgraph::visit::NodeCount,
    G: petgraph::visit::IntoEdgesDirected,
    G::NodeId: Eq + std::hash::Hash,
    G::EdgeId: Eq + std::hash::Hash,
    TargetColl: FromIterator<(Option<G::EdgeId>, G::NodeId)>,
{
    use indexmap::IndexSet;
    let mut visited: IndexSet<(Option<G::EdgeId>, G::NodeId)> = IndexSet::from_iter([(None, from)]);
    let mut stack = vec![graph.edges_directed(from, Outgoing)];

    std::iter::from_fn(move || {
        while let Some(children) = stack.last_mut() {
            if let Some(edge) = children.next() {
                let child = edge.target();
                if child == to {
                    let path = visited
                        .iter()
                        .cloned()
                        .chain([(Some(edge.id()), to)])
                        .collect::<TargetColl>();
                    return Some(path);
                } else if !visited.contains(&(Some(edge.id()), child)) {
                    visited.insert((Some(edge.id()), child));
                    stack.push(graph.edges_directed(child, Outgoing));
                }
            } else {
                stack.pop();
                visited.pop();
            }
        }
        None
    })
}

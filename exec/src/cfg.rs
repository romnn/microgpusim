use crate::model::{MemInstruction, ThreadInstruction};
use petgraph::prelude::*;

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
    #[must_use]
    pub fn id(&self) -> usize {
        match self {
            Self::Branch { id, .. } | Self::Reconverge { id, .. } => *id,
        }
    }

    #[inline]
    #[must_use]
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
    #[must_use]
    pub fn id(&self) -> usize {
        match self {
            Self::Branch(id) | Self::Reconverge(id) => *id,
        }
    }
}

#[allow(clippy::match_same_arms)]
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
                        .copied()
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

pub fn build_control_flow_graph(
    thread_instructions: &[ThreadInstruction],
    super_cfg: &mut CFG,
) -> (ThreadCFG, (NodeIndex, NodeIndex)) {
    let mut thread_cfg = ThreadCFG::new();
    let mut took_branch = false;

    let super_cfg_root_node_idx = super_cfg.add_unique_node(Node::Branch(0));
    let mut last_super_cfg_node_idx = super_cfg_root_node_idx;

    let thread_cfg_root_node_idx = thread_cfg.add_node(TraceNode::Branch {
        id: 0,
        instructions: vec![],
    });
    let mut last_thread_cfg_node_idx = thread_cfg_root_node_idx;
    let mut current_instructions = Vec::new();

    for thread_instruction in thread_instructions {
        match thread_instruction {
            ThreadInstruction::Nop => unreachable!(),
            ThreadInstruction::TookBranch(_branch_id) => {
                took_branch = true;
            }
            ThreadInstruction::Branch(branch_id) => {
                took_branch = false;
                {
                    let super_node_idx = super_cfg.add_unique_node(Node::Branch(*branch_id));
                    super_cfg.add_unique_edge(last_super_cfg_node_idx, super_node_idx, true);
                    last_super_cfg_node_idx = super_node_idx;
                }
                {
                    let node_idx = thread_cfg.add_node(TraceNode::Branch {
                        id: *branch_id,
                        instructions: std::mem::take(&mut current_instructions),
                    });
                    // assert!(!matches!(
                    //     thread_cfg[last_thread_cfg_node_idx],
                    //     cfg::TraceNode::Branch { .. }
                    // ));
                    thread_cfg.add_edge(last_thread_cfg_node_idx, node_idx, true);
                    last_thread_cfg_node_idx = node_idx;
                }
            }
            ThreadInstruction::Reconverge(branch_id) => {
                {
                    let super_node_idx = super_cfg.add_unique_node(Node::Reconverge(*branch_id));
                    super_cfg.add_unique_edge(last_super_cfg_node_idx, super_node_idx, took_branch);
                    last_super_cfg_node_idx = super_node_idx;
                }
                {
                    let node_idx = thread_cfg.add_node(TraceNode::Reconverge {
                        id: *branch_id,
                        instructions: std::mem::take(&mut current_instructions),
                    });
                    thread_cfg.add_edge(last_thread_cfg_node_idx, node_idx, took_branch);
                    // assert_eq!(
                    //     thread_cfg[last_thread_cfg_node_idx].id(),
                    //     thread_cfg[node_idx].id()
                    // );
                    last_thread_cfg_node_idx = node_idx;
                }
            }
            ThreadInstruction::Access(access) => {
                current_instructions.push(access.clone());
            }
        }
    }

    // reconverge branch 0
    let super_cfg_sink_node_idx = super_cfg.add_unique_node(Node::Reconverge(0));
    super_cfg.add_unique_edge(last_super_cfg_node_idx, super_cfg_sink_node_idx, true);

    let thread_cfg_sink_node_idx = thread_cfg.add_node(TraceNode::Reconverge {
        id: 0,
        instructions: std::mem::take(&mut current_instructions),
    });
    thread_cfg.add_edge(last_thread_cfg_node_idx, thread_cfg_sink_node_idx, true);
    (
        thread_cfg,
        (thread_cfg_root_node_idx, thread_cfg_sink_node_idx),
    )
}

pub fn format_control_flow_path<'a, G>(
    graph: &'a G,
    path: &'a [(Option<G::EdgeId>, G::NodeId)],
) -> impl Iterator<Item = String> + 'a
where
    G: petgraph::data::DataMap,
    G: petgraph::visit::Data,
    <G as petgraph::visit::Data>::NodeWeight: std::fmt::Display,
    <G as petgraph::visit::Data>::EdgeWeight: std::fmt::Display,
{
    path.iter().flat_map(move |(edge_idx, node_idx)| {
        let mut parts: Vec<String> = vec![format!("{}", graph.node_weight(*node_idx).unwrap())];
        if let Some(edge_idx) = edge_idx {
            parts.push(format!("--{}-->", graph.edge_weight(*edge_idx).unwrap()));
        }
        parts.into_iter().rev()
    })
}

pub fn add_missing_control_flow_edges<D, Ix>(graph: &mut petgraph::Graph<Node, bool, D, Ix>)
where
    Ix: petgraph::graph::IndexType,
    D: petgraph::EdgeType,
{
    use std::collections::HashSet;
    for node_idx in graph.node_indices() {
        let Node::Branch(branch_id) = graph[node_idx] else {
            continue;
        };
        let edges: HashSet<bool> = graph
            .edges(node_idx)
            .map(|edge| edge.weight())
            .copied()
            .collect();
        assert!(!edges.is_empty());
        assert!(edges.len() <= 2);
        let reconvergence_node_idx = graph.find_node(&Node::Reconverge(branch_id)).unwrap();
        if !edges.contains(&true) {
            graph.add_unique_edge(node_idx, reconvergence_node_idx, true);
        }
        if !edges.contains(&false) {
            graph.add_unique_edge(node_idx, reconvergence_node_idx, false);
        }
    }
}

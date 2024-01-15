use crate::model::{MemInstruction, ThreadInstruction};
use petgraph::prelude::*;

#[derive(Debug)]
pub enum TraceNode {
    Branch {
        branch_id: usize,
        id: usize,
        instructions: Vec<MemInstruction>,
    },
    Reconverge {
        branch_id: usize,
        id: usize,
        instructions: Vec<MemInstruction>,
    },
}

impl std::fmt::Display for TraceNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let branch_id = self.branch_id();
        let id = self.id();
        let num_instructions = self.instructions().len();
        match self {
            Self::Branch { .. } => write!(f, "BRA")?,
            Self::Reconverge { .. } => write!(f, "REC")?,
        }
        write!(f, "@{branch_id}@{id}")?;
        if num_instructions > 0 {
            write!(f, "[{num_instructions}]")?;
        }
        Ok(())
    }
}

impl TraceNode {
    // #[inline]
    #[must_use]
    pub fn branch_id(&self) -> usize {
        match self {
            Self::Branch { branch_id, .. } | Self::Reconverge { branch_id, .. } => *branch_id,
        }
    }

    // #[inline]
    #[must_use]
    pub fn id(&self) -> usize {
        match self {
            Self::Branch { id, .. } | Self::Reconverge { id, .. } => *id,
        }
    }

    // #[inline]
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
    Branch { id: usize, branch_id: usize },
    Reconverge { id: usize, branch_id: usize },
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Node {
    // #[inline]
    #[must_use]
    pub fn id(&self) -> usize {
        match self {
            Self::Branch { id, .. } | Self::Reconverge { id, .. } => *id,
        }
    }

    #[must_use]
    pub fn branch_id(&self) -> usize {
        match self {
            Self::Branch { branch_id, .. } | Self::Reconverge { branch_id, .. } => *branch_id,
        }
    }
}

#[allow(clippy::match_same_arms)]
impl PartialEq<TraceNode> for Node {
    fn eq(&self, other: &TraceNode) -> bool {
        match (self, other) {
            (Self::Branch { .. }, TraceNode::Branch { .. }) => {
                self.id() == other.id() && self.branch_id() == other.branch_id()
            }
            (Self::Reconverge { .. }, TraceNode::Reconverge { .. }) => {
                self.id() == other.id() && self.branch_id() == other.branch_id()
            }
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

pub trait Neighbors<N, E, Ix> {
    fn outgoing_neigbors(&self, node: NodeIndex<Ix>) -> Vec<(EdgeIndex<Ix>, NodeIndex<Ix>)>;
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

impl<N, E, D, Ix> Neighbors<N, E, Ix> for Graph<N, E, D, Ix>
where
    D: petgraph::EdgeType,
    Ix: petgraph::graph::IndexType,
    E: Ord,
{
    fn outgoing_neigbors(&self, node: NodeIndex<Ix>) -> Vec<(EdgeIndex<Ix>, NodeIndex<Ix>)> {
        let mut neighbors = Vec::new();
        let mut edges = self.neighbors_directed(node, petgraph::Outgoing).detach();

        while let Some((outgoing_edge_idx, next_node_idx)) = edges.next(self) {
            neighbors.push((outgoing_edge_idx, next_node_idx));
        }

        neighbors.sort_by_key(|(e, _)| &self[*e]);
        neighbors
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
    use std::collections::HashMap;
    let mut thread_cfg = ThreadCFG::new();

    let super_cfg_root_node_idx = super_cfg.add_unique_node(Node::Branch {
        id: 0, // there cannot be more than one source node
        branch_id: 0,
    });
    let mut last_super_cfg_node_idx = super_cfg_root_node_idx;

    let thread_cfg_root_node_idx = thread_cfg.add_node(TraceNode::Branch {
        id: 0, // there cannot be more than one source node
        branch_id: 0,
        instructions: vec![],
    });
    let mut last_thread_cfg_node_idx = thread_cfg_root_node_idx;
    let mut current_instructions = Vec::new();
    let mut branch_taken: HashMap<usize, bool> = HashMap::new();

    let mut unique_node_ids: HashMap<usize, usize> = HashMap::new();

    for thread_instruction in thread_instructions {
        match thread_instruction {
            ThreadInstruction::Nop => unreachable!(),
            ThreadInstruction::TookBranch(branch_id) => {
                branch_taken.insert(*branch_id, true);
            }
            ThreadInstruction::Branch(branch_id) => {
                let node_id = *unique_node_ids.entry(*branch_id).or_insert(0);
                let last_branch_id = thread_cfg[last_thread_cfg_node_idx].branch_id();
                let took_branch = branch_taken.get(&last_branch_id).copied().unwrap_or(false);

                {
                    let super_node_idx = super_cfg.add_unique_node(Node::Branch {
                        id: node_id,
                        branch_id: *branch_id,
                    });
                    super_cfg.add_unique_edge(last_super_cfg_node_idx, super_node_idx, took_branch);
                    last_super_cfg_node_idx = super_node_idx;
                }
                {
                    let instructions = std::mem::take(&mut current_instructions);
                    let node_idx = thread_cfg.add_node(TraceNode::Branch {
                        id: node_id,
                        branch_id: *branch_id,
                        instructions,
                    });
                    thread_cfg.add_edge(last_thread_cfg_node_idx, node_idx, took_branch);
                    last_thread_cfg_node_idx = node_idx;
                }
            }
            ThreadInstruction::Reconverge(branch_id) => {
                let node_id = unique_node_ids.get_mut(branch_id).unwrap();
                let reconverge_took_branch = branch_taken.get(&branch_id).copied().unwrap_or(false);

                {
                    let super_node_idx = super_cfg.add_unique_node(Node::Reconverge {
                        id: *node_id,
                        branch_id: *branch_id,
                    });
                    super_cfg.add_unique_edge(
                        last_super_cfg_node_idx,
                        super_node_idx,
                        reconverge_took_branch,
                    );
                    last_super_cfg_node_idx = super_node_idx;
                }
                {
                    let instructions = std::mem::take(&mut current_instructions);
                    let node_idx = thread_cfg.add_node(TraceNode::Reconverge {
                        id: *node_id,
                        branch_id: *branch_id,
                        instructions,
                    });
                    thread_cfg.add_edge(last_thread_cfg_node_idx, node_idx, reconverge_took_branch);
                    last_thread_cfg_node_idx = node_idx;
                }

                *node_id += 1;
            }
            ThreadInstruction::Access(access) => {
                current_instructions.push(access.clone());
            }
        }
    }

    // reconverge branch 0
    let super_cfg_sink_node_idx = super_cfg.add_unique_node(Node::Reconverge {
        id: 0, // there cannot be more than one sink node
        branch_id: 0,
    });
    super_cfg.add_unique_edge(last_super_cfg_node_idx, super_cfg_sink_node_idx, true);

    let thread_cfg_sink_node_idx = thread_cfg.add_node(TraceNode::Reconverge {
        id: 0, // there cannot be more than one sink node
        branch_id: 0,
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
    path: impl IntoIterator<Item = &'a (Option<G::EdgeId>, G::NodeId)> + 'a,
) -> impl Iterator<Item = String> + 'a
where
    G: petgraph::data::DataMap,
    G: petgraph::visit::Data,
    <G as petgraph::visit::Data>::NodeWeight: std::fmt::Display,
    <G as petgraph::visit::Data>::EdgeWeight: std::fmt::Display,
{
    path.into_iter().flat_map(move |(edge_idx, node_idx)| {
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
    let mut added = 0;
    for node_idx in graph.node_indices() {
        let Node::Branch{id, branch_id} = graph[node_idx] else {
            continue;
        };
        let edges: HashSet<bool> = graph
            .edges(node_idx)
            .map(|edge| edge.weight())
            .copied()
            .collect();
        assert!(!edges.is_empty());
        assert!(edges.len() <= 2);
        let reconvergence_node_idx = graph
            .find_node(&Node::Reconverge { id, branch_id })
            .unwrap();
        if !edges.contains(&true) {
            graph.add_unique_edge(node_idx, reconvergence_node_idx, true);
            added += 1;
        }
        if !edges.contains(&false) {
            graph.add_unique_edge(node_idx, reconvergence_node_idx, false);
            added += 1;
        }
    }
    log::trace!("added {} missing control flow graph edges", added);
}

pub mod visit {
    use super::{Neighbors, Node, CFG};
    use petgraph::graph::{EdgeIndex, NodeIndex};
    use std::collections::HashSet;

    pub type Path = Vec<(Option<EdgeIndex>, NodeIndex)>;

    #[derive(Debug)]
    pub struct DominatedDfs<'a> {
        dominator_stack: Vec<NodeIndex>,
        visited: HashSet<(EdgeIndex, NodeIndex)>,
        stack: Vec<(EdgeIndex, NodeIndex)>,
        path: Path,
        graph: &'a CFG,
    }

    impl<'a> DominatedDfs<'a> {
        #[must_use]
        pub fn new(graph: &'a CFG, root_node_idx: NodeIndex) -> Self {
            let mut dominator_stack = Vec::new();
            let mut stack = Vec::new();

            if let Node::Branch { .. } = graph[root_node_idx] {
                dominator_stack.push(root_node_idx);
            }

            for (outgoing_edge_idx, next_node_idx) in graph.outgoing_neigbors(root_node_idx) {
                stack.push((outgoing_edge_idx, next_node_idx));
            }

            Self {
                graph,
                dominator_stack,
                visited: HashSet::new(),
                path: Path::new(),
                stack,
            }
        }
    }

    impl<'a> Iterator for DominatedDfs<'a> {
        type Item = (EdgeIndex, NodeIndex);
        fn next(&mut self) -> Option<Self::Item> {
            let Some((edge_idx, node_idx)) = self.stack.pop() else {
                log::trace!("done");
                return None;
            };

            self.path.push((Some(edge_idx), node_idx));

            log::trace!(
                "dominator: {:?} {}",
                self.dominator_stack.last().map(|dom| &self.graph[*dom]),
                super::format_control_flow_path(self.graph, self.path.as_slice())
                    .collect::<String>(),
            );

            self.visited.insert((edge_idx, node_idx));
            self.path.clear();

            match &self.graph[node_idx] {
                Node::Reconverge { branch_id, .. } => {
                    // Encountered a reconvergence point.
                    //
                    // Jump back to the last branch node to serialize other possible
                    // control flow paths.
                    let last_branch_node_idx = self.dominator_stack.last().unwrap();
                    if *branch_id > 0 {
                        // When returning early from the kernel, we may not reconverge the last
                        // branch node.
                        // The last branch node id must either match the current reconvergence
                        // branch id, or the branch id is 0, which represent source and sink of the
                        // CFG.
                        assert_eq!(self.graph[*last_branch_node_idx].branch_id(), *branch_id);
                    }

                    self.stack.clear();

                    // continue at the last branch node
                    for (outgoing_edge_idx, next_node_idx) in
                        self.graph.outgoing_neigbors(*last_branch_node_idx)
                    {
                        if self.visited.contains(&(outgoing_edge_idx, next_node_idx)) {
                            continue;
                        }
                        self.stack.push((outgoing_edge_idx, next_node_idx));
                    }

                    if self.stack.is_empty() {
                        log::trace!(
                            "last branch node {} completed: proceed from {}",
                            self.graph[*last_branch_node_idx],
                            self.graph[node_idx],
                        );
                        // continue from this reconvergence point
                        self.dominator_stack.pop();

                        for (outgoing_edge_idx, next_node_idx) in
                            self.graph.outgoing_neigbors(node_idx)
                        {
                            if self.visited.contains(&(outgoing_edge_idx, next_node_idx)) {
                                continue;
                            }

                            self.stack.push((outgoing_edge_idx, next_node_idx));
                        }
                    } else {
                        log::trace!(
                            "jump back to last branch node {}",
                            self.graph[*last_branch_node_idx],
                        );
                    }

                    // do not add children of reconvergence node until all control flow paths
                    // reached convergence.
                }
                Node::Branch { .. } => {
                    // add new branch and reconvergence point on the stack
                    self.dominator_stack.push(node_idx);

                    // continue dfs
                    let mut has_children = false;
                    for (outgoing_edge_idx, next_node_idx) in self.graph.outgoing_neigbors(node_idx)
                    {
                        if self.visited.contains(&(outgoing_edge_idx, next_node_idx)) {
                            continue;
                        }
                        has_children = true;
                        self.stack.push((outgoing_edge_idx, next_node_idx));
                    }
                    if !has_children {
                        self.path.pop();
                    }
                }
            }

            Some((edge_idx, node_idx))
        }
    }
}

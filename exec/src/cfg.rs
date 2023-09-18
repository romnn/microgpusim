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
        let id = self.id();
        let num_instructions = self.instructions().len();
        match self {
            Self::Branch { .. } => write!(f, "BRA")?,
            Self::Reconverge { .. } => write!(f, "REC")?,
        }
        write!(f, "@{}", id)?;
        if num_instructions > 0 {
            write!(f, "[{}]", num_instructions)?;
        }
        Ok(())
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

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
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
{
    fn outgoing_neigbors(&self, node: NodeIndex<Ix>) -> Vec<(EdgeIndex<Ix>, NodeIndex<Ix>)> {
        let mut neighbors = Vec::new();
        let mut edges = self.neighbors_directed(node, petgraph::Outgoing).detach();

        while let Some((outgoing_edge_idx, next_node_idx)) = edges.next(&self) {
            neighbors.push((outgoing_edge_idx, next_node_idx));
        }
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
    let mut thread_cfg = ThreadCFG::new();
    // let mut took_branch = false;

    let super_cfg_root_node_idx = super_cfg.add_unique_node(Node::Branch(0));
    let mut last_super_cfg_node_idx = super_cfg_root_node_idx;

    let thread_cfg_root_node_idx = thread_cfg.add_node(TraceNode::Branch {
        id: 0,
        instructions: vec![],
    });
    let mut last_thread_cfg_node_idx = thread_cfg_root_node_idx;
    let mut current_instructions = Vec::new();
    let mut branch_taken = std::collections::HashMap::new();

    for thread_instruction in thread_instructions {
        match thread_instruction {
            ThreadInstruction::Nop => unreachable!(),
            ThreadInstruction::TookBranch(branch_id) => {
                branch_taken.insert(branch_id, true);
                // took_branch = true;
            }
            ThreadInstruction::Branch(branch_id) => {
                // took_branch = false;
                // let reconverge_took_branch = branch_taken.get(branch_id).copied().unwrap_or(false);
                let last_branch_id = thread_cfg[last_thread_cfg_node_idx].id();
                let took_branch = branch_taken.get(&last_branch_id).copied().unwrap_or(false);
                // let took_branch = match &thread_cfg[last_thread_cfg_node_idx] {
                //     TraceNode::Branch { id, .. } => branch_taken.get(id).copied().unwrap_or(false),
                //     TraceNode::Reconverge { .. } => true,
                // };

                {
                    let super_node_idx = super_cfg.add_unique_node(Node::Branch(*branch_id));
                    super_cfg.add_unique_edge(last_super_cfg_node_idx, super_node_idx, took_branch);
                    last_super_cfg_node_idx = super_node_idx;
                }
                {
                    let node_idx = thread_cfg.add_node(TraceNode::Branch {
                        id: *branch_id,
                        instructions: std::mem::take(&mut current_instructions),
                    });
                    thread_cfg.add_edge(last_thread_cfg_node_idx, node_idx, took_branch);
                    last_thread_cfg_node_idx = node_idx;
                }
            }
            ThreadInstruction::Reconverge(branch_id) => {
                // let reconverge_took_branch = true;
                // let reconverge_took_branch = branch_taken.get(branch_id).copied().unwrap_or(false);
                // let reconverge_took_branch = match &thread_cfg[last_thread_cfg_node_idx] {
                //     TraceNode::Branch { id, .. } => branch_taken.get(&id).copied().unwrap_or(false),
                //     TraceNode::Reconverge { .. } => true,
                // };
                // let last_branch_id = thread_cfg[last_thread_cfg_node_idx].id();
                let reconverge_took_branch = branch_taken.get(&branch_id).copied().unwrap_or(false);

                {
                    let super_node_idx = super_cfg.add_unique_node(Node::Reconverge(*branch_id));
                    super_cfg.add_unique_edge(
                        last_super_cfg_node_idx,
                        super_node_idx,
                        reconverge_took_branch,
                    );
                    last_super_cfg_node_idx = super_node_idx;
                }
                {
                    let node_idx = thread_cfg.add_node(TraceNode::Reconverge {
                        id: *branch_id,
                        instructions: std::mem::take(&mut current_instructions),
                    });
                    thread_cfg.add_edge(last_thread_cfg_node_idx, node_idx, reconverge_took_branch);
                    last_thread_cfg_node_idx = node_idx;
                }
            }
            ThreadInstruction::Access(access) => {
                current_instructions.push(access.clone());
            }
        }
    }

    // for edge in thread_cfg.edge_references() {
    //     if
    //     if let Node::Branch() = thread_cfg[node_idx]
    // }
    // for node_idx in thread_cfg.node_indices() {
    //     // if let Node::Branch() = thread_cfg[node_idx]
    // }
    // let reconverge_took_branch = branch_taken.get(branch_id).copied().unwrap_or(false);

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

#[allow(dead_code)]
pub mod visit {
    use super::{Neighbors, Node, UniqueGraph, CFG};
    use petgraph::graph::{EdgeIndex, NodeIndex};
    use std::collections::{HashSet, VecDeque};

    #[derive(Debug)]
    pub struct DominatedBfs<'a> {
        dominator_stack: VecDeque<NodeIndex>,
        visited: HashSet<(EdgeIndex, NodeIndex)>,
        stack: VecDeque<(EdgeIndex, NodeIndex)>,
        path: VecDeque<(Option<EdgeIndex>, NodeIndex)>,
        limit: Option<usize>,

        graph: &'a CFG,
    }

    impl<'a> DominatedBfs<'a> {
        pub fn new(graph: &'a CFG, root_node_idx: NodeIndex) -> Self {
            let mut dominator_stack = VecDeque::new();
            assert!(matches!(graph[root_node_idx], Node::Branch(0)));
            dominator_stack.push_front(root_node_idx);

            Self {
                graph,
                dominator_stack,
                visited: HashSet::new(),
                path: VecDeque::new(),
                stack: VecDeque::new(),
                limit: None,
            }
        }
    }

    impl<'a> Iterator for DominatedBfs<'a> {
        type Item = (EdgeIndex, NodeIndex);
        fn next(&mut self) -> Option<Self::Item> {
            // if let Some((edge_idx, node_idx)) = self.stack.pop_front() {
            if let Some((edge_idx, node_idx)) = self.stack.pop_back() {
                // add to path
                self.path.push_back((Some(edge_idx), node_idx));
                println!(
                    "current path: {} stack: {:?}",
                    super::format_control_flow_path(self.graph, self.path.make_contiguous())
                        .collect::<Vec<_>>()
                        .join(""),
                    self.stack
                        .iter()
                        .map(|(e, n)| (self.graph[*e], self.graph[*n].to_string()))
                        .collect::<Vec<_>>(),
                );

                self.visited.insert((edge_idx, node_idx));

                match &self.graph[node_idx] {
                    Node::Reconverge(..) => {
                        if self.dominator_stack.contains(&node_idx) {
                            // stop here, never go beyond the domninators reconvergence point
                            // println!(
                            //     "current: dominator={:?} \t taken={} --> {:?} \t STOP: found reconvergence point",
                            //     self.graph[reconvergence_node_idx],
                            //     self.graph[edge_idx],
                            //     self.graph[node_idx],
                            //     // active_mask,
                            // );
                            self.path.pop_back();
                            self.path.pop_back();
                            return Some((edge_idx, node_idx));
                        }
                    }
                    Node::Branch(branch_id) => {
                        // must handle new branch
                        let reconvergence_point = Node::Reconverge(*branch_id);
                        let reconvergence_node_idx =
                            self.graph.find_node(&reconvergence_point).unwrap();
                        self.dominator_stack.push_back(reconvergence_node_idx);
                    }
                }

                // let reconvergence_node_idx: Option<petgraph::graph::NodeIndex> =
                //     self.dominator_stack.front().copied();

                // println!(
                //     "current: dominator={:?} \t taken={} --> {:?}",
                //     reconvergence_node_idx.map(|idx| &self.graph[idx]),
                //     self.graph[edge_idx],
                //     self.graph[node_idx],
                //     // active_mask,
                // );

                let mut has_children = false;

                let mut outgoing_neigbors = self.graph.outgoing_neigbors(node_idx);
                outgoing_neigbors.sort_by_key(|(e, _)| self.graph[*e]);
                // self.path.pop_back();
                for (outgoing_edge_idx, next_node_idx) in outgoing_neigbors {
                    if self.visited.contains(&(outgoing_edge_idx, next_node_idx)) {
                        continue;
                    }

                    has_children = true;
                    self.stack.push_back((outgoing_edge_idx, next_node_idx));
                }

                // let mut edges = self
                //     .graph
                //     .neighbors_directed(node_idx, petgraph::Outgoing)
                //     .detach();
                //
                // // while let Some((outgoing_edge_idx, next_node_idx)) = edges.next(&self.graph) {
                // while let Some((outgoing_edge_idx, next_node_idx)) = edges.next(&self.graph) {
                //     // println!(
                //     //     "pushing branch \t {:?} --> taken={} --> {:?}",
                //     //     self.graph[node_idx],
                //     //     self.graph[outgoing_edge_idx],
                //     //     self.graph[next_node_idx],
                //     // );
                //     if self.visited.contains(&(outgoing_edge_idx, next_node_idx)) {
                //         continue;
                //     }
                //
                //     has_children = true;
                //     self.stack.push_back((outgoing_edge_idx, next_node_idx));
                //     // self.stack.push_front((edge_idx, node_idx));
                //     // break;
                // }

                if !has_children {
                    self.path.pop_back();
                }
                return Some((edge_idx, node_idx));
            } else {
                // stack is empty, proceed with parent denominator
                // if let Some(reconvergence_node_idx) = self.dominator_stack.pop_front() {
                if let Some(reconvergence_node_idx) = self.dominator_stack.pop_back() {
                    println!("all reconverged {:?}", self.graph[reconvergence_node_idx]);

                    let mut outgoing_neigbors =
                        self.graph.outgoing_neigbors(reconvergence_node_idx);
                    outgoing_neigbors.sort_by_key(|(e, _)| self.graph[*e]);

                    // self.path.pop_back();
                    for (outgoing_edge_idx, next_node_idx) in outgoing_neigbors {
                        if self.visited.contains(&(outgoing_edge_idx, next_node_idx)) {
                            continue;
                        }

                        self.stack.push_back((outgoing_edge_idx, next_node_idx));
                        // self.stack.push_back((outgoing_edge_idx, next_node_idx));
                        // self.stack.push_front(child);
                    }

                    // let mut edges = self
                    //     .graph
                    //     .neighbors_directed(reconvergence_node_idx, petgraph::Outgoing)
                    //     .detach();
                    //
                    // self.path.pop_back();
                    //
                    // while let Some((outgoing_edge_idx, next_node_idx)) = edges.next(&self.graph) {
                    //     if self.visited.contains(&(outgoing_edge_idx, next_node_idx)) {
                    //         continue;
                    //     }
                    //
                    //     self.stack.push_back((outgoing_edge_idx, next_node_idx));
                    //     // self.stack.push_back((outgoing_edge_idx, next_node_idx));
                    //     // self.stack.push_front(child);
                    // }
                    // // self.path = [(None, reconvergence_node_idx)].into_iter().collect();
                }

                if self.stack.is_empty() {
                    // done
                    println!("done");
                    return None;
                }

                if let Some(ref mut limit) = self.limit {
                    *limit = limit.checked_sub(1).unwrap_or(0);
                    assert!(*limit != 0, "WARNING: limit reached");
                }

                return self.next();
            }
        }
    }
}

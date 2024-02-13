use crate::model::{Instruction, ThreadInstruction};
use petgraph::prelude::*;

#[derive(Debug)]
pub enum ThreadNode {
    Branch {
        branch_id: usize,
        id: usize,
        instructions: Vec<Instruction>,
    },
    Reconverge {
        branch_id: usize,
        id: usize,
        instructions: Vec<Instruction>,
    },
}

impl std::fmt::Display for ThreadNode {
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

impl ThreadNode {
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
    pub fn instructions(&self) -> &[Instruction] {
        match self {
            Self::Branch { instructions, .. } | Self::Reconverge { instructions, .. } => {
                instructions
            }
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarpNode {
    Branch { id: usize, branch_id: usize },
    Reconverge { id: usize, branch_id: usize },
}

impl std::fmt::Display for WarpNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl WarpNode {
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
impl PartialEq<ThreadNode> for WarpNode {
    fn eq(&self, other: &ThreadNode) -> bool {
        match (self, other) {
            (Self::Branch { .. }, ThreadNode::Branch { .. }) => {
                self.id() == other.id() && self.branch_id() == other.branch_id()
            }
            (Self::Reconverge { .. }, ThreadNode::Reconverge { .. }) => {
                self.id() == other.id() && self.branch_id() == other.branch_id()
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Edge(bool);

impl Edge {
    pub fn took_branch(&self) -> bool {
        self.0
    }

    pub fn taken(&self) -> bool {
        self.0
    }
}

impl std::fmt::Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub type WarpCFG = petgraph::graph::DiGraph<WarpNode, Edge>;
pub type ThreadCFG = petgraph::graph::DiGraph<ThreadNode, Edge>;

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
                }
                if !visited.contains(&(Some(edge.id()), child)) {
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
    warp_cfg: &mut WarpCFG,
) -> (ThreadCFG, (NodeIndex, NodeIndex)) {
    use std::collections::HashMap;
    let mut thread_cfg = ThreadCFG::new();

    let warp_cfg_root_node_idx = warp_cfg.add_unique_node(WarpNode::Branch {
        id: 0, // there cannot be more than one source node
        branch_id: 0,
    });
    let mut last_warp_cfg_node_idx = warp_cfg_root_node_idx;

    let thread_cfg_root_node_idx = thread_cfg.add_node(ThreadNode::Branch {
        id: 0, // there cannot be more than one source node
        branch_id: 0,
        instructions: vec![],
    });
    let mut last_thread_cfg_node_idx = thread_cfg_root_node_idx;
    let mut current_instructions: Vec<Instruction> = Vec::new();
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
                    let super_node_idx = warp_cfg.add_unique_node(WarpNode::Branch {
                        id: node_id,
                        branch_id: *branch_id,
                    });
                    warp_cfg.add_unique_edge(
                        last_warp_cfg_node_idx,
                        super_node_idx,
                        Edge(took_branch),
                    );
                    last_warp_cfg_node_idx = super_node_idx;
                }
                {
                    let instructions = std::mem::take(&mut current_instructions);
                    let node_idx = thread_cfg.add_node(ThreadNode::Branch {
                        id: node_id,
                        branch_id: *branch_id,
                        instructions,
                    });
                    thread_cfg.add_edge(last_thread_cfg_node_idx, node_idx, Edge(took_branch));
                    last_thread_cfg_node_idx = node_idx;
                }
            }
            ThreadInstruction::Reconverge(branch_id) => {
                let node_id = unique_node_ids.get_mut(branch_id).unwrap();
                let reconverge_took_branch = branch_taken.get(&branch_id).copied().unwrap_or(false);

                {
                    let super_node_idx = warp_cfg.add_unique_node(WarpNode::Reconverge {
                        id: *node_id,
                        branch_id: *branch_id,
                    });
                    warp_cfg.add_unique_edge(
                        last_warp_cfg_node_idx,
                        super_node_idx,
                        Edge(reconverge_took_branch),
                    );
                    last_warp_cfg_node_idx = super_node_idx;
                }
                {
                    let instructions = std::mem::take(&mut current_instructions);
                    let node_idx = thread_cfg.add_node(ThreadNode::Reconverge {
                        id: *node_id,
                        branch_id: *branch_id,
                        instructions,
                    });
                    thread_cfg.add_edge(
                        last_thread_cfg_node_idx,
                        node_idx,
                        Edge(reconverge_took_branch),
                    );
                    last_thread_cfg_node_idx = node_idx;
                }

                *node_id += 1;
            }
            ThreadInstruction::Access(access) => {
                current_instructions.push(Instruction::Memory(access.clone()));
            }
            ThreadInstruction::Barrier => {
                current_instructions.push(Instruction::Barrier);
            }
        }
    }

    // reconverge branch 0
    let warp_cfg_sink_node_idx = warp_cfg.add_unique_node(WarpNode::Reconverge {
        id: 0, // there cannot be more than one sink node
        branch_id: 0,
    });
    warp_cfg.add_unique_edge(last_warp_cfg_node_idx, warp_cfg_sink_node_idx, Edge(true));

    let thread_cfg_sink_node_idx = thread_cfg.add_node(ThreadNode::Reconverge {
        id: 0, // there cannot be more than one sink node
        branch_id: 0,
        instructions: std::mem::take(&mut current_instructions),
    });
    thread_cfg.add_edge(
        last_thread_cfg_node_idx,
        thread_cfg_sink_node_idx,
        Edge(true),
    );
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

pub fn add_missing_control_flow_edges<D, Ix>(graph: &mut petgraph::Graph<WarpNode, bool, D, Ix>)
where
    Ix: petgraph::graph::IndexType,
    D: petgraph::EdgeType,
{
    use std::collections::HashSet;
    let mut added = 0;
    for node_idx in graph.node_indices() {
        let WarpNode::Branch { id, branch_id } = graph[node_idx] else {
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
            .find_node(&WarpNode::Reconverge { id, branch_id })
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
    use super::{Neighbors, WarpCFG, WarpNode};
    use petgraph::graph::{EdgeIndex, NodeIndex};
    use std::collections::HashSet;

    pub type Path = Vec<(Option<EdgeIndex>, NodeIndex)>;

    #[derive(Debug)]
    pub struct DominatedDfs<'a> {
        dominator_stack: Vec<NodeIndex>,
        visited: HashSet<(EdgeIndex, NodeIndex)>,
        stack: Vec<(EdgeIndex, NodeIndex)>,
        path: Path,
        graph: &'a WarpCFG,
    }

    impl<'a> DominatedDfs<'a> {
        #[must_use]
        pub fn new(graph: &'a WarpCFG, root_node_idx: NodeIndex) -> Self {
            let mut dominator_stack = Vec::new();
            let mut stack = Vec::new();

            if let WarpNode::Branch { .. } = graph[root_node_idx] {
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
                WarpNode::Reconverge { branch_id, .. } => {
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
                WarpNode::Branch { .. } => {
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

pub mod render {
    use std::path::Path;

    pub trait Label {
        fn label(&self) -> String;
    }

    impl Label for crate::cfg::WarpNode {
        fn label(&self) -> String {
            match self {
                crate::cfg::WarpNode::Branch { id, branch_id } => {
                    format!("BRANCH {branch_id}\n#{id}")
                }
                crate::cfg::WarpNode::Reconverge { id, branch_id } => {
                    format!("RECONVERGE {branch_id}\n#{id}")
                }
            }
        }
    }

    impl Label for crate::cfg::ThreadNode {
        fn label(&self) -> String {
            match self {
                crate::cfg::ThreadNode::Branch {
                    id,
                    branch_id,
                    instructions,
                } => {
                    format!("BRANCH {branch_id}\n#{id}\n{} instr", instructions.len())
                }
                crate::cfg::ThreadNode::Reconverge {
                    id,
                    branch_id,
                    instructions,
                } => {
                    format!(
                        "RECONVERGE {branch_id}\n#{id}\n{} instr",
                        instructions.len()
                    )
                }
            }
        }
    }

    impl Label for crate::cfg::Edge {
        fn label(&self) -> String {
            format!("took branch = {}", self.0)
        }
    }

    pub trait Render {
        /// Render graph as an svg image.
        ///
        /// # Errors
        /// If writing to the specified output path fails.
        fn render_to(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error>;
    }

    impl<N, E, D, Ix> Render for petgraph::Graph<N, E, D, Ix>
    where
        D: petgraph::EdgeType,
        Ix: petgraph::graph::IndexType,
        E: Label,
        N: Label,
    {
        fn render_to(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
            use layout::adt::dag::NodeHandle;
            use layout::backends::svg::SVGWriter;
            use layout::core::{self, base::Orientation, color::Color, style};
            use layout::std_shapes::shapes;
            use layout::topo::layout::VisualGraph;
            use petgraph::graph::NodeIndex;
            use std::collections::HashMap;
            use std::io::{BufWriter, Write};

            fn node_circle<N>(node: &N) -> shapes::Element
            where
                N: Label,
            {
                let node_style = style::StyleAttr {
                    line_color: Color::new(0x0000_00FF),
                    line_width: 2,
                    fill_color: Some(Color::new(0xB4B3_B2FF)),
                    rounded: 0,
                    font_size: 15,
                };
                let size = core::geometry::Point { x: 100.0, y: 100.0 };
                shapes::Element::create(
                    shapes::ShapeKind::Circle(node.label()),
                    node_style,
                    Orientation::TopToBottom,
                    size,
                )
            }

            let mut graph = VisualGraph::new(Orientation::TopToBottom);
            let mut handles: HashMap<NodeIndex<Ix>, NodeHandle> = HashMap::new();

            // add nodes
            for node_idx in self.node_indices() {
                let node = self.node_weight(node_idx).unwrap();
                handles
                    .entry(node_idx)
                    .or_insert_with(|| graph.add_node(node_circle(node)));
            }

            // add edges
            for edge_idx in self.edge_indices() {
                // let Some((src_node, dest_node)) = self.edge_endpoints(edge_idx) else {
                let (src_node_idx, dest_node_idx) = self.edge_endpoints(edge_idx).unwrap();
                // let src_handle = *handles
                //     .entry(src_node)
                //     .or_insert_with(|| graph.add_node(node(self.node_weight(src_node).unwrap())));
                //
                // let dest_handle = *handles
                //     .entry(src_node)
                //     .or_insert_with(|| graph.add_node(node(self.node_weight(dest_node).unwrap())));

                // let src_handle = handles[&src_node_idx];
                // let dest_handle = handles[&dest_node_idx];

                let edge = self.edge_weight(edge_idx).unwrap();
                let arrow = shapes::Arrow {
                    start: shapes::LineEndKind::None,
                    end: shapes::LineEndKind::Arrow,
                    line_style: style::LineStyleKind::Normal,
                    text: edge.label(),
                    look: style::StyleAttr {
                        line_color: Color::new(0x0000_00FF),
                        line_width: 2,
                        fill_color: Some(Color::new(0xB4B3_B2FF)),
                        rounded: 0,
                        font_size: 15,
                    },
                    src_port: None,
                    dst_port: None,
                };
                eprintln!(
                    "edge {} from {:?} to {:?} => {:?} to {:?}",
                    edge.label(),
                    src_node_idx,
                    dest_node_idx,
                    handles[&src_node_idx],
                    handles[&dest_node_idx]
                );
                graph.add_edge(arrow, handles[&src_node_idx], handles[&dest_node_idx]);
            }

            // https://docs.rs/layout-rs/latest/src/layout/backends/svg.rs.html#200
            let mut backend = SVGWriter::new();
            let debug_mode = false;
            let disable_opt = false;
            let disable_layout = false;
            graph.do_it(debug_mode, disable_opt, disable_layout, &mut backend);
            let content = backend.finalize();

            let file = std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(path.as_ref())?;
            let mut writer = BufWriter::new(file);
            writer.write_all(content.as_bytes())?;
            Ok(())
        }
    }
}

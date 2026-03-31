const exampleProfile = {
  name: "샘플 학생",
  free_text:
    "수학적 사고로 알고리즘 문제를 해결하는 것이 즐겁고, 데이터 분석 프로젝트와 연구 프로젝트에 몰입합니다. 데이터 과학 전공과 데이터 사이언티스트 진로에 관심이 있습니다.",
  skills: ["데이터 분석", "머신러닝", "프로그래밍", "의사소통"],
  activities: ["Data Analysis Project", "Research Project"],
  career_interests: ["Data Science", "Artificial Intelligence"],
  major_interests: ["Data Science", "Computer Science"],
  traits: ["Investigative", "Innovation"],
  preferred_job_zones: [4, 5],
  top_k: 5,
};

const dom = {
  resultCaption: document.querySelector("#result-caption"),
  summaryLoading: document.querySelector("#summary-loading"),
  graph: document.querySelector("#ontology-graph"),
  graphStage: document.querySelector("#graph-stage"),
  graphCanvas: document.querySelector("#graph-canvas"),
  graphModalBar: document.querySelector("#graph-modal-bar"),
  graphModalClose: document.querySelector("#graph-modal-close"),
  graphModalBackdrop: document.querySelector("#graph-modal-backdrop"),
  graphEmpty: document.querySelector("#graph-empty"),
  graphTooltip: document.querySelector("#graph-tooltip"),
  graphCaption: document.querySelector("#graph-caption"),
  graphFocusSummary: document.querySelector("#graph-focus-summary"),
  graphStats: document.querySelector("#graph-stats"),
  zoomOut: document.querySelector("#zoom-out"),
  zoomReset: document.querySelector("#zoom-reset"),
  zoomIn: document.querySelector("#zoom-in"),
  graphExpand: document.querySelector("#graph-expand"),
  zoomLevel: document.querySelector("#zoom-level"),
  rightPanel: document.querySelector(".right-panel"),
  selectionSummary: document.querySelector("#selection-summary"),
  selectionDetails: document.querySelector("#selection-details"),
  reportOverview: document.querySelector("#report-overview"),
  actionSummary: document.querySelector("#action-summary"),
  actionResults: document.querySelector("#action-results"),
  actionActivityInput: document.querySelector("#action-activity-input"),
  actionSuggestions: document.querySelector("#action-suggestions"),
  nodeSearch: document.querySelector("#node-search"),
  impactSort: document.querySelector("#impact-sort"),
  quickCareerFocus: document.querySelector("#quick-focus-careers"),
  quickMajorFocus: document.querySelector("#quick-focus-majors"),
  quickActivityFocus: document.querySelector("#quick-focus-activities"),
  quickSkillFocus: document.querySelector("#quick-focus-skills"),
};

const viewState = {
  report: null,
  workspace: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  nodeFilters: new Set(["student", "skill", "activity", "major", "career", "trait"]),
  searchText: "",
  entityCache: new Map(),
  latestSimulation: null,
  graphZoom: 1,
  graphExpanded: false,
  renderedGraph: { nodes: [], edges: [] },
};

const GRAPH_BASE_WIDTH = 1200;
const GRAPH_BASE_HEIGHT = 860;
const GRAPH_ZOOM_MIN = 0.8;
const GRAPH_ZOOM_MAX = 2.2;
const GRAPH_ZOOM_STEP = 0.2;

const ENTITY_LABELS = {
  student: "학생",
  skill: "역량",
  activity: "활동",
  major: "전공",
  career: "진로",
  trait: "성향",
  knowledge: "지식",
};

const RELATION_LABELS = {
  hasSkill: "보유 역량",
  inferredSkill: "추론 역량",
  experiencedActivity: "활동 경험",
  hasTrait: "성향",
  hasInterestIn: "관심",
  requiresSkill: "요구 역량",
  developsSkill: "역량 형성",
  leadsTo: "진로 연결",
  supportsCareer: "진로 지원",
  supportsMajor: "전공 지원",
  hasPrerequisite: "브리지 전제",
};

const COLUMN_LABELS = {
  0: "학생",
  1: "현재 상태",
  2: "브리지",
  3: "전공·활동",
  4: "진로",
};

const TYPE_COLORS = {
  student: "#f8fafc",
  skill: "#7dd3fc",
  activity: "#22c55e",
  major: "#f59e0b",
  career: "#c084fc",
  trait: "#94a3b8",
  knowledge: "#f472b6",
};

function parseList(value) {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function collectProfile() {
  const zones = [...document.querySelectorAll('input[name="job-zone"]:checked')].map((item) =>
    Number(item.value)
  );

  return {
    name: document.querySelector("#student-name").value.trim() || "이름 미입력 학생",
    free_text: document.querySelector("#free-text").value.trim(),
    skills: parseList(document.querySelector("#skills").value),
    activities: parseList(document.querySelector("#activities").value),
    career_interests: parseList(document.querySelector("#career-interests").value),
    major_interests: parseList(document.querySelector("#major-interests").value),
    traits: parseList(document.querySelector("#traits").value),
    preferred_job_zones: zones,
    top_k: 5,
  };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function fillForm(profile) {
  document.querySelector("#student-name").value = profile.name || "";
  document.querySelector("#free-text").value = profile.free_text || "";
  document.querySelector("#skills").value = (profile.skills || [])
    .map((skill) => (typeof skill === "string" ? skill : skill.name))
    .join(", ");
  document.querySelector("#activities").value = (profile.activities || []).join(", ");
  document.querySelector("#career-interests").value = (profile.career_interests || []).join(", ");
  document.querySelector("#major-interests").value = (profile.major_interests || []).join(", ");
  document.querySelector("#traits").value = (profile.traits || []).join(", ");
  document.querySelectorAll('input[name="job-zone"]').forEach((box) => {
    box.checked = (profile.preferred_job_zones || []).includes(Number(box.value));
  });
}

function resetForm() {
  fillForm({
    name: "",
    free_text: "",
    skills: [],
    activities: [],
    career_interests: [],
    major_interests: [],
    traits: [],
    preferred_job_zones: [],
  });
  clearWorkspace();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderChips(items, secondary = false) {
  if (!items || items.length === 0) {
    return `<p class="empty-state">정보가 없습니다.</p>`;
  }
  return `<div class="chip-wrap">${items
    .map(
      (item) =>
        `<span class="chip ${secondary ? "secondary" : ""}">${escapeHtml(item)}</span>`
    )
    .join("")}</div>`;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(3);
}

function fieldLabel(value) {
  const labels = {
    free_text: "자유 서술 입력",
    skills: "보유 역량",
    activities: "활동 경험",
    career_interests: "관심 진로",
    major_interests: "관심 전공",
    traits: "성향 단서",
  };
  return labels[value] || value;
}

function activeNodeId() {
  return viewState.hoveredNodeId || viewState.selectedNodeId;
}

function setGraphEmpty(isEmpty) {
  dom.graphEmpty.hidden = !isEmpty;
  dom.graphEmpty.classList.toggle("hidden", !isEmpty);
}

function syncGraphCanvas() {
  const width = Math.round(GRAPH_BASE_WIDTH * viewState.graphZoom);
  const height = Math.round(GRAPH_BASE_HEIGHT * viewState.graphZoom);
  dom.graphCanvas.style.width = `${width}px`;
  dom.graphCanvas.style.height = `${height}px`;
  dom.zoomLevel.textContent = `${Math.round(viewState.graphZoom * 100)}%`;
  dom.graphExpand.textContent = viewState.graphExpanded ? "모달 닫기" : "모달 열기";
  dom.graphStage.classList.toggle("modalized", viewState.graphExpanded);
  dom.graphModalBar.hidden = !viewState.graphExpanded;
  dom.graphModalBar.classList.toggle("hidden", !viewState.graphExpanded);
  dom.graphModalBackdrop.hidden = !viewState.graphExpanded;
  dom.graphModalBackdrop.classList.toggle("hidden", !viewState.graphExpanded);
  document.body.classList.toggle("graph-modal-open", viewState.graphExpanded);
}

function setGraphZoom(nextZoom) {
  const stage = dom.graphStage;
  const previousWidth = dom.graphCanvas.offsetWidth || GRAPH_BASE_WIDTH;
  const previousHeight = dom.graphCanvas.offsetHeight || GRAPH_BASE_HEIGHT;
  const centerRatioX = (stage.scrollLeft + stage.clientWidth / 2) / previousWidth;
  const centerRatioY = (stage.scrollTop + stage.clientHeight / 2) / previousHeight;

  viewState.graphZoom = clamp(nextZoom, GRAPH_ZOOM_MIN, GRAPH_ZOOM_MAX);
  syncGraphCanvas();

  const nextWidth = dom.graphCanvas.offsetWidth || GRAPH_BASE_WIDTH;
  const nextHeight = dom.graphCanvas.offsetHeight || GRAPH_BASE_HEIGHT;
  stage.scrollLeft = Math.max(0, nextWidth * centerRatioX - stage.clientWidth / 2);
  stage.scrollTop = Math.max(0, nextHeight * centerRatioY - stage.clientHeight / 2);
}

function hideTooltip() {
  dom.graphTooltip.classList.add("hidden");
  dom.graphTooltip.innerHTML = "";
}

function closeGraphModal() {
  if (!viewState.graphExpanded) {
    return;
  }
  viewState.graphExpanded = false;
  syncGraphCanvas();
}

function scrollSelectionPanelToTop() {
  dom.rightPanel.scrollTo({ top: 0, behavior: "auto" });
}

function positionTooltip(event) {
  if (dom.graphTooltip.classList.contains("hidden")) {
    return;
  }
  const stageRect = dom.graphStage.getBoundingClientRect();
  const tooltip = dom.graphTooltip;
  const rawLeft = event.clientX - stageRect.left + dom.graphStage.scrollLeft + 16;
  const rawTop = event.clientY - stageRect.top + dom.graphStage.scrollTop + 16;
  const maxLeft =
    dom.graphStage.scrollLeft + dom.graphStage.clientWidth - tooltip.offsetWidth - 12;
  const maxTop =
    dom.graphStage.scrollTop + dom.graphStage.clientHeight - tooltip.offsetHeight - 12;
  tooltip.style.left = `${Math.max(dom.graphStage.scrollLeft + 12, Math.min(rawLeft, maxLeft))}px`;
  tooltip.style.top = `${Math.max(dom.graphStage.scrollTop + 12, Math.min(rawTop, maxTop))}px`;
}

function clearWorkspace() {
  viewState.report = null;
  viewState.workspace = null;
  viewState.selectedNodeId = null;
  viewState.hoveredNodeId = null;
  viewState.latestSimulation = null;
  viewState.renderedGraph = { nodes: [], edges: [] };
  viewState.entityCache.clear();
  viewState.graphZoom = 1;
  viewState.graphExpanded = false;
  syncGraphCanvas();
  dom.resultCaption.textContent = "아직 분석을 실행하지 않았습니다.";
  dom.graphCaption.textContent =
    "학생 상태를 분석하면 중심 그래프가 생성되고, 노드 선택에 따라 맥락이 재정렬됩니다.";
  dom.graphFocusSummary.textContent = "포커스가 없을 때는 전체 관계망이 보입니다.";
  dom.graphStats.textContent = "노드와 관계 수가 여기에 표시됩니다.";
  dom.graph.innerHTML = "";
  setGraphEmpty(true);
  hideTooltip();
  resetSelectionSummary();
  dom.selectionDetails.classList.add("empty-state");
  dom.selectionDetails.innerHTML =
    "그래프 노드를 선택하면 의미 맥락, 연결 관계, 준비도 경로가 이 영역에 표시됩니다.";
  dom.reportOverview.classList.add("empty-state");
  dom.reportOverview.innerHTML = "분석 후 대표 진로, 전공, 활동이 운영 표 형태로 정리됩니다.";
  dom.actionResults.classList.add("empty-state");
  dom.actionResults.innerHTML = "아직 시뮬레이션 결과가 없습니다.";
  dom.actionSummary.textContent =
    "활동 노드를 가상으로 주입해 진로와 역량 구조가 어떻게 재배열되는지 확인할 수 있습니다.";
  dom.actionActivityInput.value = "";
  dom.actionSuggestions.innerHTML = "분석 후 추천 활동이 여기 표시됩니다.";
  dom.quickCareerFocus.innerHTML = "분석 후 주요 진로 노드를 바로 선택할 수 있습니다.";
  dom.quickMajorFocus.innerHTML = "";
  dom.quickActivityFocus.innerHTML = "";
  dom.quickSkillFocus.innerHTML = "";
}

async function loadSummary() {
  const response = await fetch("/ontology/summary");
  const summary = await response.json();
  const entityCount = Object.values(summary.entity_counts).reduce((acc, value) => acc + value, 0);
  const relationCount = Object.values(summary.relation_counts).reduce((acc, value) => acc + value, 0);
  dom.summaryLoading.textContent = `엔티티 ${entityCount}개 · 관계 ${relationCount}개`;
}

function getNodeById(nodeId) {
  return (viewState.workspace?.nodes || []).find((node) => node.id === nodeId) || null;
}

function resolveNodeIdByLabel(label, entityType = null) {
  const normalizedLabel = String(label || "").trim().toLowerCase();
  const node = (viewState.workspace?.nodes || []).find((item) => {
    if (entityType && item.entity_type !== entityType) {
      return false;
    }
    return item.label.trim().toLowerCase() === normalizedLabel;
  });
  return node?.id || null;
}

function getFilteredGraph() {
  if (!viewState.workspace) {
    return { nodes: [], edges: [] };
  }

  let nodes = viewState.workspace.nodes.filter((node) => viewState.nodeFilters.has(node.entity_type));
  const search = viewState.searchText.trim().toLowerCase();
  if (search) {
    const matchedIds = new Set(
      nodes
        .filter((node) => node.label.toLowerCase().includes(search))
        .map((node) => node.id)
    );
    const expandedIds = new Set(matchedIds);
    viewState.workspace.edges.forEach((edge) => {
      if (matchedIds.has(edge.source) || matchedIds.has(edge.target)) {
        expandedIds.add(edge.source);
        expandedIds.add(edge.target);
      }
    });
    nodes = nodes.filter((node) => expandedIds.has(node.id));
  }

  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = viewState.workspace.edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)
  );
  return { nodes, edges };
}

function buildAdjacency(edges) {
  const adjacency = new Map();
  edges.forEach((edge) => {
    if (!adjacency.has(edge.source)) {
      adjacency.set(edge.source, new Set());
    }
    if (!adjacency.has(edge.target)) {
      adjacency.set(edge.target, new Set());
    }
    adjacency.get(edge.source).add(edge.target);
    adjacency.get(edge.target).add(edge.source);
  });
  return adjacency;
}

function computeFocusContext(nodes, edges, focusNodeId) {
  const nodeIds = new Set(nodes.map((node) => node.id));
  if (!focusNodeId || !nodeIds.has(focusNodeId)) {
    return {
      focusNodes: new Set(nodeIds),
      focusEdges: new Set(edges.map((edge) => `${edge.source}|${edge.target}|${edge.relation}`)),
      relatedNodes: new Set(),
    };
  }

  const adjacency = buildAdjacency(edges);
  const focusNodes = new Set([focusNodeId]);
  const relatedNodes = new Set();
  const queue = [{ id: focusNodeId, depth: 0 }];
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || current.depth >= 2) {
      continue;
    }
    const neighbors = adjacency.get(current.id) || new Set();
    neighbors.forEach((neighborId) => {
      if (!focusNodes.has(neighborId)) {
        focusNodes.add(neighborId);
        relatedNodes.add(neighborId);
        queue.push({ id: neighborId, depth: current.depth + 1 });
      }
    });
  }

  const focusEdges = new Set();
  edges.forEach((edge) => {
    if (focusNodes.has(edge.source) && focusNodes.has(edge.target)) {
      focusEdges.add(`${edge.source}|${edge.target}|${edge.relation}`);
    }
  });

  return { focusNodes, focusEdges, relatedNodes };
}

function nodePriority(node) {
  const priorities = {
    student: 0,
    career_primary: 1,
    career_secondary: 2,
    recommended_major: 3,
    recommended_activity: 4,
    bridge_skill: 5,
    gap_skill: 6,
    declared_skill: 7,
    inferred_skill: 8,
    matched_skill: 9,
    experienced_activity: 10,
    trait: 11,
  };
  return priorities[node.status] ?? 99;
}

function nodeColor(node) {
  if (node.status === "bridge_skill") {
    return "#f59e0b";
  }
  if (node.status === "gap_skill") {
    return "#ef4444";
  }
  return TYPE_COLORS[node.entity_type] || "#94a3b8";
}

function nodeDimensions(node) {
  const baseWidth = node.entity_type === "skill" ? 130 : 170;
  const width = Math.min(Math.max(baseWidth, node.label.length * 7 + 60), 240);
  const height = node.entity_type === "skill" ? 44 : 52;
  return { width, height };
}

function relationLabel(relation) {
  return RELATION_LABELS[relation] || relation;
}

function truncateLabel(value, maxLength = 24) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}…` : value;
}

function updateGraphCopy(nodes, edges) {
  const activeId = activeNodeId();
  const visibleActiveId = nodes.some((node) => node.id === activeId) ? activeId : null;
  const filterSummary = [...viewState.nodeFilters].map((type) => ENTITY_LABELS[type]).join(", ");
  dom.graphStats.textContent = `노드 ${nodes.length}개 · 관계 ${edges.length}개 · 필터 ${filterSummary}`;

  if (!visibleActiveId) {
    dom.graphFocusSummary.textContent =
      "학생을 중심으로 현재 상태, 브리지 역량, 전공·활동, 진로 결과가 계층형으로 정렬되어 있습니다.";
    dom.graphCaption.textContent =
      "노드를 선택하거나 마우스를 올리면 관련 관계만 남고 나머지 구조는 희미해집니다.";
    return;
  }

  const node = getNodeById(visibleActiveId);
  if (!node) {
    dom.graphFocusSummary.textContent = "현재 포커스 노드가 필터에서 숨겨져 있습니다.";
    dom.graphCaption.textContent =
      "필터를 조정하거나 다른 노드를 선택하면 다시 의미 경로를 확인할 수 있습니다.";
    return;
  }

  const verb = viewState.hoveredNodeId ? "마우스를 올린" : "선택한";
  dom.graphFocusSummary.textContent =
    `${verb} ${node.label} 주변의 2단계 관계만 강조했습니다. 직접 연결과 브리지 연결이 동시에 보입니다.`;
  dom.graphCaption.textContent =
    `${node.label} 중심 맥락입니다. 관련 없는 노드는 약화되어 의미 경로만 남습니다.`;
}

function updateGraphEmphasis() {
  const { nodes, edges } = viewState.renderedGraph;
  const candidateFocusId = activeNodeId();
  const focusId = nodes.some((node) => node.id === candidateFocusId) ? candidateFocusId : null;
  const { focusEdges, relatedNodes } = computeFocusContext(nodes, edges, focusId);

  dom.graph.querySelectorAll(".graph-edge").forEach((path) => {
    const edgeKey = path.dataset.edgeKey || "";
    path.classList.remove("focused", "related", "dimmed");
    if (!focusId) {
      return;
    }
    path.classList.add(focusEdges.has(edgeKey) ? "focused" : "dimmed");
  });

  dom.graph.querySelectorAll(".graph-node").forEach((group) => {
    const nodeId = group.dataset.nodeId || "";
    const node = nodes.find((item) => item.id === nodeId);
    const color = node ? nodeColor(node) : "#94a3b8";
    const isFocused = nodeId === focusId;
    const isRelated = relatedNodes.has(nodeId);
    group.classList.remove("focused", "related", "dimmed");
    if (focusId) {
      group.classList.add(isFocused ? "focused" : isRelated ? "related" : "dimmed");
    }
    const rect = group.querySelector("rect");
    if (rect) {
      rect.setAttribute("fill", isFocused ? `${color}26` : isRelated ? `${color}12` : "#10161f");
    }
  });

  updateGraphCopy(nodes, edges);
}

function showGraphTooltip(node, event) {
  const context = workspaceContext(node.id);
  const degree = context.incoming.length + context.outgoing.length;
  dom.graphTooltip.innerHTML = `
    <strong>${escapeHtml(node.label)}</strong>
    <p>${escapeHtml(ENTITY_LABELS[node.entity_type] || node.entity_type)} · 연결 ${degree}개${
      node.score !== null && node.score !== undefined ? ` · 점수 ${formatScore(node.score)}` : ""
    }</p>
  `;
  dom.graphTooltip.classList.remove("hidden");
  positionTooltip(event);
}

function renderGraph() {
  const { nodes, edges } = getFilteredGraph();
  viewState.renderedGraph = { nodes, edges };
  dom.graph.innerHTML = "";
  hideTooltip();

  if (!nodes.length) {
    setGraphEmpty(true);
    dom.graphStats.textContent = "현재 필터에서 보이는 노드가 없습니다.";
    dom.graphFocusSummary.textContent = "조건에 맞는 노드가 없어 의미 경로를 표시할 수 없습니다.";
    return;
  }

  setGraphEmpty(false);

  const width = GRAPH_BASE_WIDTH;
  const height = GRAPH_BASE_HEIGHT;
  dom.graph.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const paddingTop = 80;
  const paddingBottom = 40;
  const paddingLeft = 90;
  const paddingRight = 100;
  const maxColumn = Math.max(...nodes.map((node) => node.column), 4);
  const columnWidth = (width - paddingLeft - paddingRight) / maxColumn;
  const positions = new Map();

  const columns = new Map();
  nodes.forEach((node) => {
    if (!columns.has(node.column)) {
      columns.set(node.column, []);
    }
    columns.get(node.column).push(node);
  });

  for (let column = 0; column <= maxColumn; column += 1) {
    const columnNodes = (columns.get(column) || []).sort((left, right) => {
      const scoreDelta = (right.score || 0) - (left.score || 0);
      if (Math.abs(scoreDelta) > 0.0001) {
        return scoreDelta;
      }
      const priorityDelta = nodePriority(left) - nodePriority(right);
      if (priorityDelta !== 0) {
        return priorityDelta;
      }
      return left.label.localeCompare(right.label);
    });

    const x = paddingLeft + columnWidth * column;
    const innerHeight = height - paddingTop - paddingBottom;
    const yStep = innerHeight / (columnNodes.length + 1);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(x));
    label.setAttribute("y", "36");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "graph-column-label");
    label.textContent = COLUMN_LABELS[column] || `컬럼 ${column}`;
    dom.graph.appendChild(label);

    columnNodes.forEach((node, index) => {
      const { width: nodeWidth, height: nodeHeight } = nodeDimensions(node);
      positions.set(node.id, {
        x,
        y: paddingTop + yStep * (index + 1),
        width: nodeWidth,
        height: nodeHeight,
      });
    });
  }

  edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) {
      return;
    }

    const sourceRight = source.x + source.width / 2;
    const targetLeft = target.x - target.width / 2;
    const startX = sourceRight;
    const endX = targetLeft;
    const startY = source.y;
    const endY = target.y;
    const controlOffset = Math.max((endX - startX) * 0.45, 50);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute(
      "d",
      `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`
    );
    path.setAttribute("class", `graph-edge status-${edge.status}`);
    path.dataset.edgeKey = `${edge.source}|${edge.target}|${edge.relation}`;
    path.setAttribute("stroke-width", String(Math.max(1.2, edge.weight * 2.4)));
    dom.graph.appendChild(path);
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    if (!position) {
      return;
    }
    const color = nodeColor(node);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "graph-node");
    group.dataset.nodeId = node.id;
    group.addEventListener("click", () => {
      setSelectedNode(node.id);
    });
    group.addEventListener("mouseenter", (event) => {
      viewState.hoveredNodeId = node.id;
      updateGraphEmphasis();
      showGraphTooltip(node, event);
    });
    group.addEventListener("mousemove", (event) => {
      positionTooltip(event);
    });
    group.addEventListener("mouseleave", () => {
      if (viewState.hoveredNodeId === node.id) {
        viewState.hoveredNodeId = null;
      }
      hideTooltip();
      updateGraphEmphasis();
    });

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(position.x - position.width / 2));
    rect.setAttribute("y", String(position.y - position.height / 2));
    rect.setAttribute("width", String(position.width));
    rect.setAttribute("height", String(position.height));
    rect.setAttribute("fill", "#10161f");
    rect.setAttribute("stroke", color);
    group.appendChild(rect);

    const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
    title.setAttribute("x", String(position.x - position.width / 2 + 10));
    title.setAttribute("y", String(position.y - 4));
    title.setAttribute("class", "node-label");
    title.textContent = truncateLabel(node.label, node.entity_type === "career" ? 26 : 22);
    group.appendChild(title);

    const meta = document.createElementNS("http://www.w3.org/2000/svg", "text");
    meta.setAttribute("x", String(position.x - position.width / 2 + 10));
    meta.setAttribute("y", String(position.y + 14));
    meta.setAttribute("class", "node-meta");
    meta.textContent = `${ENTITY_LABELS[node.entity_type] || node.entity_type}${
      node.score !== null && node.score !== undefined ? ` · ${formatScore(node.score)}` : ""
    }`;
    group.appendChild(meta);

    dom.graph.appendChild(group);
  });

  updateGraphEmphasis();
}

function focusButtons(items) {
  if (!items || items.length === 0) {
    return `<p class="empty-state">표시할 노드가 없습니다.</p>`;
  }
  return items
    .map(
      (item) => `
        <button class="focus-button" type="button" data-focus-node="${escapeHtml(item.id)}">
          ${escapeHtml(item.label)}
        </button>
      `
    )
    .join("");
}

function renderQuickFocus() {
  if (!viewState.report || !viewState.workspace) {
    return;
  }
  const nodes = viewState.workspace.nodes;
  const topCareerNodes = viewState.report.top_careers
    .map((career) => nodes.find((node) => node.id === career.career_id))
    .filter(Boolean);
  const topMajorNodes = viewState.report.top_majors
    .map((major) => nodes.find((node) => node.id === major.major_id))
    .filter(Boolean);
  const topActivityNodes = viewState.report.top_activities
    .map((activity) => nodes.find((node) => node.id === activity.activity_id))
    .filter(Boolean);
  const topSkillNodes = nodes
    .filter((node) => node.entity_type === "skill")
    .sort((left, right) => (right.score || 0) - (left.score || 0))
    .slice(0, 6);

  dom.quickCareerFocus.innerHTML = focusButtons(topCareerNodes);
  dom.quickMajorFocus.innerHTML = focusButtons(topMajorNodes);
  dom.quickActivityFocus.innerHTML = focusButtons(topActivityNodes);
  dom.quickSkillFocus.innerHTML = focusButtons(topSkillNodes);
}

function relationshipRows(relationships) {
  if (!relationships.length) {
    return `<p class="empty-state">가시적인 관계가 없습니다.</p>`;
  }
  return `
    <table class="detail-table">
      <thead>
        <tr>
          <th>관계</th>
          <th>상대 노드</th>
          <th>가중치</th>
        </tr>
      </thead>
      <tbody>
        ${relationships
          .map(
            (item) => `
              <tr class="clickable-row" data-focus-node="${escapeHtml(item.nodeId)}">
                <td>${escapeHtml(item.relation)}</td>
                <td>${escapeHtml(item.label)}</td>
                <td class="mono">${formatScore(item.weight)}</td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function workspaceContext(nodeId) {
  if (!viewState.workspace) {
    return { incoming: [], outgoing: [] };
  }
  const incoming = [];
  const outgoing = [];
  viewState.workspace.edges.forEach((edge) => {
    if (edge.source === nodeId) {
      const target = getNodeById(edge.target);
      if (target) {
        outgoing.push({
          relation: relationLabel(edge.relation),
          label: target.label,
          nodeId: target.id,
          weight: edge.weight,
        });
      }
    }
    if (edge.target === nodeId) {
      const source = getNodeById(edge.source);
      if (source) {
        incoming.push({
          relation: relationLabel(edge.relation),
          label: source.label,
          nodeId: source.id,
          weight: edge.weight,
        });
      }
    }
  });
  incoming.sort((left, right) => right.weight - left.weight);
  outgoing.sort((left, right) => right.weight - left.weight);
  return { incoming, outgoing };
}

function resetSelectionSummary() {
  dom.selectionSummary.classList.add("empty-state");
  dom.selectionSummary.innerHTML = "노드를 선택하면 핵심 요약이 이 영역에 고정됩니다.";
}

function renderSelectionSummary({
  title,
  entityType,
  status,
  score = null,
  description = "",
  incomingCount = 0,
  outgoingCount = 0,
}) {
  dom.selectionSummary.classList.remove("empty-state");
  dom.selectionSummary.innerHTML = `
    <strong>${escapeHtml(title)}</strong>
    <div>${escapeHtml(ENTITY_LABELS[entityType] || entityType)} · ${escapeHtml(status || "focus")}${
      score !== null && score !== undefined ? ` · ${formatScore(score)}` : ""
    }</div>
    <div>${escapeHtml(description || "선택한 노드의 핵심 맥락을 유지해서 보여줍니다.")}</div>
    <div class="meta-row">
      <span class="type-pill">들어옴 ${incomingCount}</span>
      <span class="type-pill">나감 ${outgoingCount}</span>
    </div>
  `;
}

function compactAttributes(attributes) {
  if (!attributes) {
    return [];
  }
  const preferredKeys = [
    "job_zone",
    "soc_code",
    "profile_kind",
    "post_count",
    "record_count",
    "job_zone_reference",
  ];
  return preferredKeys
    .filter((key) => Object.prototype.hasOwnProperty.call(attributes, key))
    .map((key) => {
      const rawValue = attributes[key];
      if (typeof rawValue === "object") {
        return `${key}: ${JSON.stringify(rawValue)}`;
      }
      return `${key}: ${rawValue}`;
    });
}

async function fetchEntityDetails(entityId) {
  if (viewState.entityCache.has(entityId)) {
    return viewState.entityCache.get(entityId);
  }
  const response = await fetch(`/ontology/entities/${encodeURIComponent(entityId)}`);
  if (!response.ok) {
    return null;
  }
  const data = await response.json();
  viewState.entityCache.set(entityId, data);
  return data;
}

function renderStudentDetails() {
  const report = viewState.report;
  if (!report) {
    return;
  }
  const readiness = report.student_readiness;
  renderSelectionSummary({
    title: report.student_name,
    entityType: "student",
    status: "primary_focus",
    score: null,
    description: readiness.explanation,
    incomingCount: 0,
    outgoingCount: readiness.priority_bridge_skills.length,
  });
  dom.selectionDetails.classList.remove("empty-state");
  dom.selectionDetails.innerHTML = `
    <article class="detail-card">
      <div class="detail-header">
        <div>
          <h3>${escapeHtml(report.student_name)}</h3>
          <p class="panel-note">학생 · 기본 포커스</p>
        </div>
        <span class="type-pill">Job Zone ${escapeHtml(readiness.current_job_zone)}</span>
      </div>
      <p>${escapeHtml(report.student_summary)}</p>
      <p class="panel-note">${escapeHtml(readiness.explanation)}</p>
    </article>
    <article class="detail-card">
      <p class="list-title">해석된 입력 엔티티</p>
      ${renderChips(
        report.interpreted_state.profile_interpretations
          .slice(0, 10)
          .map((item) => `${fieldLabel(item.source_field)}: ${item.entity_name}`),
        true
      )}
    </article>
    <article class="detail-card">
      <p class="list-title">다음 브리지 역량</p>
      ${renderChips(readiness.priority_bridge_skills)}
    </article>
  `;
}

async function renderSelectionDetails() {
  if (!viewState.report || !viewState.workspace || !viewState.selectedNodeId) {
    resetSelectionSummary();
    dom.selectionDetails.classList.add("empty-state");
    dom.selectionDetails.innerHTML =
      "그래프 노드를 선택하면 의미 맥락, 연결 관계, 준비도 경로가 이 영역에 표시됩니다.";
    return;
  }

  if (viewState.selectedNodeId === "workspace:student") {
    renderStudentDetails();
    return;
  }

  const node = getNodeById(viewState.selectedNodeId);
  if (!node) {
    return;
  }

  const details = await fetchEntityDetails(node.id);
  const context = workspaceContext(node.id);
  const outgoing = context.outgoing.slice(0, 8);
  const incoming = context.incoming.slice(0, 8);
  const entity = details?.entity || null;
  const attributeLines = compactAttributes(entity?.attributes || {});

  renderSelectionSummary({
    title: node.label,
    entityType: node.entity_type,
    status: node.status,
    score: node.score,
    description:
      entity?.description || "관계 구조상 현재 포커스에 직접 연결되는 핵심 엔티티입니다.",
    incomingCount: context.incoming.length,
    outgoingCount: context.outgoing.length,
  });

  dom.selectionDetails.classList.remove("empty-state");
  dom.selectionDetails.innerHTML = `
    <article class="detail-card">
      <div class="detail-header">
        <div>
          <h3>${escapeHtml(node.label)}</h3>
          <p class="panel-note">${escapeHtml(ENTITY_LABELS[node.entity_type] || node.entity_type)} · ${escapeHtml(node.status)}</p>
        </div>
        <span class="type-pill">${node.score !== null && node.score !== undefined ? formatScore(node.score) : "정보"}</span>
      </div>
      ${
        entity?.description
          ? `<p>${escapeHtml(entity.description)}</p>`
          : `<p class="panel-note">명시적 설명이 없어서 관계 구조를 기준으로 해석합니다.</p>`
      }
      ${
        attributeLines.length
          ? `<div class="detail-card">${renderChips(attributeLines, true)}</div>`
          : ""
      }
    </article>
    <article class="detail-card">
      <p class="list-title">들어오는 관계</p>
      ${relationshipRows(incoming)}
    </article>
    <article class="detail-card">
      <p class="list-title">나가는 관계</p>
      ${relationshipRows(outgoing)}
    </article>
  `;

  if (node.entity_type === "activity" && !dom.actionActivityInput.value.trim()) {
    dom.actionActivityInput.value = node.label;
  }
}

function reportCareerRows(report) {
  return report.top_careers
    .map(
      (career, index) => `
        <tr class="clickable-row" data-focus-node="${escapeHtml(career.career_id)}">
          <td>${index + 1}</td>
          <td>${escapeHtml(career.career_name)}</td>
          <td class="mono">${career.job_zone ?? "-"}</td>
          <td class="mono">${formatScore(career.score)}</td>
          <td class="mono">${career.readiness_plan.readiness_gap ?? "-"}</td>
        </tr>
      `
    )
    .join("");
}

function reportNamedRows(items, kind) {
  if (!items.length) {
    return `<p class="empty-state">구조 결과가 없습니다.</p>`;
  }
  return `
    <table class="summary-table">
      <thead>
        <tr>
          <th>${kind}</th>
          <th>점수</th>
          <th>핵심 축</th>
        </tr>
      </thead>
      <tbody>
        ${items
          .map((item) => {
            const nodeId = item.major_id || item.activity_id;
            const name = item.major_name || item.activity_name;
            const signals = item.key_skills || item.focus_skills || [];
            return `
              <tr class="clickable-row" data-focus-node="${escapeHtml(nodeId)}">
                <td>${escapeHtml(name)}</td>
                <td class="mono">${formatScore(item.score)}</td>
                <td>${escapeHtml(signals.slice(0, 2).join(", ") || "-")}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderReportOverview() {
  if (!viewState.report) {
    return;
  }
  const report = viewState.report;
  dom.reportOverview.classList.remove("empty-state");
  dom.reportOverview.innerHTML = `
    <article class="detail-card">
      <p class="list-title">학생 의미 상태</p>
      <p>${escapeHtml(report.student_summary)}</p>
      <p class="panel-note">${escapeHtml(report.decision_principle)}</p>
    </article>
    <article class="detail-card">
      <p class="list-title">상위 진로</p>
      <table class="summary-table">
        <thead>
          <tr>
            <th>순위</th>
            <th>진로</th>
            <th>Zone</th>
            <th>점수</th>
            <th>간극</th>
          </tr>
        </thead>
        <tbody>${reportCareerRows(report)}</tbody>
      </table>
    </article>
    <article class="detail-card">
      <p class="list-title">권장 전공</p>
      ${reportNamedRows(report.top_majors, "전공")}
    </article>
    <article class="detail-card">
      <p class="list-title">권장 활동</p>
      ${reportNamedRows(report.top_activities, "활동")}
    </article>
  `;
}

function renderActionSuggestions() {
  const activities = viewState.report?.top_activities || [];
  if (!activities.length) {
    dom.actionSuggestions.innerHTML = "분석 후 추천 활동이 여기 표시됩니다.";
    return;
  }
  dom.actionSuggestions.innerHTML = activities
    .map(
      (activity) => `
        <button class="focus-button" type="button" data-fill-activity="${escapeHtml(activity.activity_name)}" data-focus-node="${escapeHtml(activity.activity_id)}">
          ${escapeHtml(activity.activity_name)}
        </button>
      `
    )
    .join("");
  if (!dom.actionActivityInput.value.trim()) {
    dom.actionActivityInput.value = activities[0].activity_name;
  }
}

function sortCareerImpacts(items) {
  const sortMode = dom.impactSort.value;
  const copy = [...items];
  if (sortMode === "name_asc") {
    copy.sort((left, right) => left.career_name.localeCompare(right.career_name));
    return copy;
  }
  if (sortMode === "rank_best") {
    copy.sort((left, right) => (left.simulated_rank || 999) - (right.simulated_rank || 999));
    return copy;
  }
  copy.sort((left, right) => right.score_delta - left.score_delta);
  return copy;
}

function renderSimulationCareerTable(items) {
  if (!items.length) {
    return `<p class="empty-state">눈에 띄는 진로 변화가 없습니다.</p>`;
  }
  return `
    <table class="simulation-table">
      <thead>
        <tr>
          <th>진로</th>
          <th>기준</th>
          <th>시뮬레이션</th>
          <th>증감</th>
          <th>순위</th>
        </tr>
      </thead>
      <tbody>
        ${sortCareerImpacts(items)
          .map((item) => {
            const nodeId = resolveNodeIdByLabel(item.career_name, "career") || "";
            return `
              <tr class="clickable-row" ${nodeId ? `data-focus-node="${escapeHtml(nodeId)}"` : ""}>
                <td>${escapeHtml(item.career_name)}</td>
                <td class="mono">${formatScore(item.baseline_score)}</td>
                <td class="mono">${formatScore(item.simulated_score)}</td>
                <td class="mono ${item.score_delta >= 0 ? "delta-positive" : "delta-negative"}">
                  ${item.score_delta >= 0 ? "+" : ""}${item.score_delta_percent.toFixed(1)}%p
                </td>
                <td class="mono">${item.simulated_rank ?? "-"}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderSimulationSkillTable(items) {
  if (!items.length) {
    return `<p class="empty-state">눈에 띄는 역량 변화가 없습니다.</p>`;
  }
  return `
    <table class="simulation-table">
      <thead>
        <tr>
          <th>역량</th>
          <th>기준</th>
          <th>시뮬레이션</th>
          <th>증감</th>
        </tr>
      </thead>
      <tbody>
        ${items
          .slice()
          .sort((left, right) => right.score_delta - left.score_delta)
          .map((item) => {
            const nodeId = resolveNodeIdByLabel(item.skill_name, "skill") || "";
            return `
              <tr class="clickable-row" ${nodeId ? `data-focus-node="${escapeHtml(nodeId)}"` : ""}>
                <td>${escapeHtml(item.skill_name)}</td>
                <td class="mono">${formatScore(item.baseline_score)}</td>
                <td class="mono">${formatScore(item.simulated_score)}</td>
                <td class="mono delta-positive">+${(item.score_delta * 100).toFixed(1)}%p</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderActionSimulation(report) {
  viewState.latestSimulation = report;
  dom.actionResults.classList.remove("empty-state");
  dom.actionResults.innerHTML = `
    <article class="detail-card">
      <p class="list-title">미래 피드백</p>
      <ul class="plain-list">${report.future_feedback
        .map((item) => `<li>${escapeHtml(item)}</li>`)
        .join("")}</ul>
    </article>
    <article class="detail-card">
      <p class="list-title">진로 영향 표</p>
      ${renderSimulationCareerTable(report.career_impacts)}
    </article>
    <article class="detail-card">
      <p class="list-title">역량 영향 표</p>
      ${renderSimulationSkillTable(report.skill_impacts)}
    </article>
  `;
}

async function runAnalysis() {
  const payload = collectProfile();
  dom.resultCaption.textContent = "온톨로지 서브그래프를 생성하고 있습니다.";

  const response = await fetch("/analysis/workspace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    dom.resultCaption.textContent = "구조 분석 중 문제가 발생했습니다.";
    return;
  }

  const analysis = await response.json();
  viewState.report = analysis.report;
  viewState.workspace = analysis.workspace;
  viewState.hoveredNodeId = null;
  viewState.selectedNodeId =
    analysis.workspace.default_focus_node_id || analysis.report.top_careers?.[0]?.career_id || null;
  viewState.latestSimulation = null;
  dom.resultCaption.textContent = analysis.report.epistemic_note;

  renderQuickFocus();
  renderGraph();
  renderReportOverview();
  renderActionSuggestions();
  await renderSelectionDetails();
  dom.actionSummary.textContent =
    "활동을 가상 주입하면 그래프 아래 영향 표가 즉시 재정렬되고, 관련 활동 노드를 다시 추적할 수 있습니다.";
}

async function runActionSimulation() {
  const activityName = dom.actionActivityInput.value.trim();
  if (!activityName) {
    dom.actionSummary.textContent = "먼저 시뮬레이션할 활동명을 입력해 주세요.";
    return;
  }

  dom.actionSummary.textContent = "가상 활동 노드를 주입하고 영향 구조를 재계산하고 있습니다.";

  const response = await fetch("/actions/simulate-activity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile: collectProfile(),
      activity_name: activityName,
    }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    dom.actionSummary.textContent = body.detail || "시뮬레이션 중 문제가 발생했습니다.";
    return;
  }

  const report = await response.json();
  dom.actionSummary.textContent = report.action_summary;
  renderActionSimulation(report);
}

function setSelectedNode(nodeId) {
  viewState.hoveredNodeId = null;
  hideTooltip();
  viewState.selectedNodeId = nodeId;
  renderGraph();
  renderSelectionDetails();
  scrollSelectionPanelToTop();
}

document.addEventListener("click", (event) => {
  const focusButton = event.target.closest("[data-focus-node]");
  if (focusButton) {
    const nodeId = focusButton.dataset.focusNode;
    if (nodeId) {
      setSelectedNode(nodeId);
    }
  }

  const fillActivityButton = event.target.closest("[data-fill-activity]");
  if (fillActivityButton) {
    dom.actionActivityInput.value = fillActivityButton.dataset.fillActivity || "";
  }
});

document.querySelector("#run-analysis").addEventListener("click", runAnalysis);
document.querySelector("#run-action-simulation").addEventListener("click", runActionSimulation);
document.querySelector("#load-example").addEventListener("click", () => fillForm(exampleProfile));
document.querySelector("#reset-form").addEventListener("click", resetForm);
document.querySelector("#clear-focus").addEventListener("click", () => {
  viewState.hoveredNodeId = null;
  viewState.selectedNodeId = null;
  hideTooltip();
  renderGraph();
  renderSelectionDetails();
  scrollSelectionPanelToTop();
});

dom.zoomOut.addEventListener("click", () => {
  setGraphZoom(viewState.graphZoom - GRAPH_ZOOM_STEP);
});

dom.zoomReset.addEventListener("click", () => {
  setGraphZoom(1);
});

dom.zoomIn.addEventListener("click", () => {
  setGraphZoom(viewState.graphZoom + GRAPH_ZOOM_STEP);
});

dom.graphExpand.addEventListener("click", () => {
  viewState.graphExpanded = !viewState.graphExpanded;
  hideTooltip();
  syncGraphCanvas();
});

dom.graphModalClose.addEventListener("click", () => {
  closeGraphModal();
});

dom.graphModalBackdrop.addEventListener("click", () => {
  closeGraphModal();
});

dom.graphStage.addEventListener("mouseleave", () => {
  if (!viewState.hoveredNodeId) {
    return;
  }
  viewState.hoveredNodeId = null;
  hideTooltip();
  updateGraphEmphasis();
});

dom.graphStage.addEventListener("scroll", () => {
  hideTooltip();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeGraphModal();
  }
});

dom.nodeSearch.addEventListener("input", (event) => {
  viewState.searchText = event.target.value || "";
  renderGraph();
});

dom.impactSort.addEventListener("change", () => {
  if (viewState.latestSimulation) {
    renderActionSimulation(viewState.latestSimulation);
  }
});

document.querySelectorAll("[data-node-filter]").forEach((input) => {
  input.addEventListener("change", (event) => {
    const value = event.target.dataset.nodeFilter;
    if (!value) {
      return;
    }
    if (event.target.checked) {
      viewState.nodeFilters.add(value);
    } else {
      viewState.nodeFilters.delete(value);
    }
    renderGraph();
  });
});

resetForm();
loadSummary();

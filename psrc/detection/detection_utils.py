from typing import List

def compute_overlap(boxA: List[float], boxB: List[float]) -> float:
  """
  Compute the overlap ratio between two bounding boxes.

  This function calculates the ratio of the intersection area of two boxes to the area of the smaller box.

  Parameters:
    boxA (List[float]): Coordinates of the first bounding box in the format [x_min, y_min, x_max, y_max].
    boxB (List[float]): Coordinates of the second bounding box in the format [x_min, y_min, x_max, y_max].

  Returns:
    float: The overlap ratio (intersection area divided by the area of the smaller box), or 0.0 if there is no
    overlap.
  """
  x_left = max(boxA[0], boxB[0])
  y_top = max(boxA[1], boxB[1])
  x_right = min(boxA[2], boxB[2])
  y_bottom = min(boxA[3], boxB[3])
  
  # No overlap
  if x_right < x_left or y_bottom < y_top:
    return 0.0

  # Compute intersection area and the area of the smaller box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)
  areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  min_area = min(areaA, areaB)

  # Prevent division by zero
  if min_area == 0:
    return 0.0

  return intersection_area / min_area

def group_cards(boxes: List[List[float]], overlap_threshold: float = 0.1) -> List[List[int]]:
  """
  Group bounding boxes based on their overlap.

  This function builds an undirected graph where each bounding box is a node. An edge is added between two nodes
  if the overlap ratio computed meets or exceeds the specified threshold. The function then identifies connected
  components in the graph using Depth-First Search (DFS), where each connected component represents a group of
  boxes that likely correspond to the same card.

  Parameters:
    boxes (List[List[float]]): A list of bounding boxes, each represented by a list of coordinates [x_min,
    y_min, x_max, y_max].
    overlap_threshold (float): The minimum overlap ratio required for two boxes to be considered connected.
    Default is 0.1.

  Returns:
    List[List[int]]: A list of groups, where each group is a list of indices corresponding to the bounding boxes
    that form a connected component.
  """
  n = len(boxes)
  graph = {i: [] for i in range(n)}
  
  # Add edges between boxes with sufficient overlap
  for i in range(n):
    for j in range(i + 1, n):
      if compute_overlap(boxes[i], boxes[j]) >= overlap_threshold:
        graph[i].append(j)
        graph[j].append(i)
  
  visited = [False] * n
  groups = []

  # Find connected components using DFS
  for i in range(n):
    if not visited[i]:
      stack = [i]
      group = []

      while stack:
        node = stack.pop()

        if not visited[node]:
          visited[node] = True
          group.append(node)
          stack.extend(graph[node])

      groups.append(group)

  return groups
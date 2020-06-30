def find_shortest_path(start_loc,end_loc,grid):
    pred = bfs(start_loc,end_loc,grid)
    crawl = end_loc
    path = [crawl]  
    while crawl != start_loc:
        path.append(pred[crawl])
        crawl = pred[crawl]
    return path

def bfs(start_loc, end_loc, grid):
    dist = {}
    pred = {}
    visited = []
    queue = []

    visited.append(start_loc)
    queue.append(start_loc)
    dist[start_loc] = 0

    while queue:
        current_loc = queue.pop(0)
        
        adjacents = findAllAdjacent(current_loc,grid)
        for adjacent in adjacents:
            if adjacent not in visited:
                visited.append(adjacent)
                dist[adjacent] = dist[current_loc] + 1
                pred[adjacent] = current_loc
                queue.append(adjacent)
                if(adjacent == end_loc): return pred

def findAllAdjacent(current_loc,grid):
    search_range = [(-1,0),(0,1),(1,0),(0,-1)] #top, right, bot, left
    adj = []
    ref_row, ref_col = current_loc
    for i in range(4):
        row = ref_row + search_range[i][0]
        col = ref_col + search_range[i][1]
        if row < 0 or col < 0: continue
        try:
            if grid[row][col] == 0:
                adj.append((row,col))
        except: #out of bounds
            continue
    return adj
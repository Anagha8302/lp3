class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

def knapsack_bb(values, weights, capacity):
    n = len(values)
    items = [Item(values[i], weights[i]) for i in range(n)]

    def bound(node, capacity, value, weight):
        if weight > capacity:
            return 0
        bound = value
        j = node + 1
        total_weight = weight
        while j < n and total_weight + items[j].weight <= capacity:
            bound += items[j].value
            total_weight += items[j].weight
            j += 1
        if j < n:
            bound += (capacity - total_weight) * (items[j].value / items[j].weight)
        return bound

    def knapsack_recursive(node, capacity, value, weight):
        nonlocal max_value
        if weight > capacity:
            return

        if value > max_value:
            max_value = value

        if node == n:
            return

        if bound(node, capacity, value, weight) <= max_value:
            return

        knapsack_recursive(node + 1, capacity, value, weight)

        knapsack_recursive(node + 1, capacity - items[node].weight, value + items[node].value, weight + items[node].weight)

    max_value = 0
    knapsack_recursive(0, capacity, 0, 0)
    return max_value

# Example usage:
if __name__ == "__main__":
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    max_value = knapsack_bb(values, weights, capacity)
    print("Maximum value:", max_value)

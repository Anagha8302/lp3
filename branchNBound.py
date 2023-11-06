class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

def knapsack_branch_and_bound(items, capacity):
    def bound(i, weight, value):
        if weight > capacity:
            return 0
        bound_value = value
        j = i
        total_weight = weight
        while j < n and total_weight + items[j].weight <= capacity:
            total_weight += items[j].weight
            bound_value += items[j].value
            j += 1
        if j < n:
            bound_value += (capacity - total_weight) * (items[j].value / items[j].weight)
        return bound_value

    def knapsack_recursive(i, weight, value):
        nonlocal max_value
        if weight <= capacity and value > max_value:
            max_value = value
        if i < n:
            if weight + items[i].weight <= capacity:
                knapsack_recursive(i + 1, weight + items[i].weight, value + items[i].value)
            if bound(i + 1, weight, value) > max_value:
                knapsack_recursive(i + 1, weight, value)

    n = len(items)
    max_value = 0
    items.sort(key=lambda x: x.value / x.weight, reverse=True)
    knapsack_recursive(0, 0, 0)
    return max_value

# Example usage
if __name__ == "__main__":
    items = [Item(10, 60), Item(20, 100), Item(30, 120)]
    capacity = 50

    max_value = knapsack_branch_and_bound(items, capacity)
    print("Maximum value in the knapsack:", max_value)

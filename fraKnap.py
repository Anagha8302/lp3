def fractional_knapsack(value, weight, capacity):
    index = list(range(len(value)))
    ratio = [v / w for v, w in zip(value, weight)]
    index.sort(key=lambda i: ratio[i], reverse=True)

    max_value = 0
    fractions = [0] * len(value)

    for i in index:
        if weight[i] <= capacity:
            fractions[i] = 1
            max_value += value[i]
            capacity -= weight[i]
        else:
            fractions[i] = capacity / weight[i]
            max_value += value[i] * (capacity / weight[i])
            break

    return max_value, fractions

n = int(input("Enter number of items: "))
value = list(map(int, input("Enter the values of the items separated by space: ").split()))
weight = list(map(int, input("Enter the positive weights of the items separated by space: ").split()))
capacity = int(input("Enter maximum weight: "))

max_value, fractions = fractional_knapsack(value, weight, capacity)
print('The maximum value of items that can be carried:', max_value)
print('The fractions in which the items should be taken:',fractions)


--------------------------------------------------------------------------------------------------------------------------------------
class item:
    def __init__(self,profit,weight):
        self.profit=profit
        self.weight=weight
def fractionalKnapsack(w,arr):
    arr.sort(key=lambda x:(x.profit/x.weight),reverse=True)
    finalVal=0.0
    for item in arr:
        if item.weight<=w:
            w-=item.weight
            finalVal+=item.profit
        else:
            finalVal+=item.profit*w/item.weight
            break
    return finalVal
if __name__=="__main__":
    w=50
    arr=[item(60,10),item(100,20),item(120,30)]
    max_val=fractionalKnapsack(w,arr)
    print(max_val)


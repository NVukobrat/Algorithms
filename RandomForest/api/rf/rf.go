package rf

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

//
// Node of the Decision Tree.
//
type Node struct {
	Index     int           `json:"index"`      // Node index.
	Value     float64       `json:"value"`      // Node value.
	LeftVal   float64       `json:"left_val"`   // Left Terminal Node value.
	RightVal  float64       `json:"right_val"`  // Right Terminal Node value.
	LeftNode  *Node         `json:"left_node"`  // Left Node.
	RightNode *Node         `json:"right_node"` // Right Node.
	Groups    [][][]float64 `json:"groups"`     // Left and Right Node groups.
}

//
// Build constructs Random Forest structure.
//
// Because Decision Trees are susceptible to high variance they are not
// ideal for larger (sometimes even smaller) datasets. To solve this,
// Random Forest algorithm is introduced.
//
// Random Forest combines multiple Decision Trees into single decision
// algorithm. Because of this approach, name Forest is appearing in the
// name. During construction of the "forest", algorithm randomly removes
// some attributes and samples from the dataset during each node construction.
// Out of this comes Random within the name.
//
//
func Build(dataset [][]float64, maxDepth, minSize, treeNum, featureNum int) []Node {
	trees := []Node{}

	for i := 0; i < treeNum; i++ {
		//TODO: Use sample of dataset, not whole for each tree.
		tree := buildDecisionTree(dataset, maxDepth, minSize, featureNum)
		trees = append(trees, tree)
	}

	return trees
}

//
// Build constructs Decision Tree by determining the best
// possible root decision node. When that is successfully compleated,
// it builds up recursively rest of the tree, node by node with
// its leafs at the end of tree. Leaf (or terminal) nodes represent
// predicted classes of the build up Decision Tree.
//
func buildDecisionTree(dataset [][]float64, maxDepth, minSize, featureNum int) Node {
	rootNode := bestSplit(dataset, featureNum)
	nodeSplit(rootNode, maxDepth, minSize, 1)

	return *rootNode
}

//
// bestSplit looks for the best possible candidate for split.
// Once best split is found, it can be used as node of the tree.
//
func bestSplit(dataset [][]float64, featureNum int) *Node {
	classes := []float64{}
	for _, row := range dataset {
		classes = appendIfMissing(classes, row[len(row)-1])
	}

	features := []int{}
	for len(features) < featureNum {
		
		index := rand.Intn(len(dataset[0]) - 1)
		if !intInSlice(index, features) {
			features = append(features, index)			
		}
	}
	//TODO: Continue
	bestIndex, bestValue, bestScore, bestGroups := 999, 999.0, 999.0, [][][]float64{}
	for index := 0; index < len(dataset[0])-1; index++ {
		for _, row := range dataset {
			groups := groupDivide(index, row[index], dataset)
			gini := giniIndex(groups, classes)
			// fmt.Printf("X%d < %.3f Gini=%.3f\n", index+1, row[index], gini)

			if gini < bestScore {
				bestIndex, bestValue, bestScore, bestGroups = index, row[index], gini, groups
			}
		}
	}

	return &Node{Index: bestIndex, Value: bestValue, Groups: bestGroups}
}

func appendIfMissing(slice []float64, i float64) []float64 {
	for _, ele := range slice {
		if ele == i {
			return slice
		}
	}
	return append(slice, i)
}

func intInSlice(a int, list []int) bool {
    for _, b := range list {
        if b == a {
            return true
        }
    }
    return false
}

//
// groupDivide splits dataset in two list of rows giving the index
// and the value of the attribute. Having those two groups, gini
// score could be calculated for given attribute.
//
func groupDivide(index int, value float64, dataset [][]float64) [][][]float64 {
	left, right := [][]float64{}, [][]float64{}

	for _, row := range dataset {
		if row[index] < value {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}

	return [][][]float64{left, right}
}

//
// nodeSplit recursively build up tree using root node as a
// reference. It uses left and right node groups
// to determine will current node will be used as new nodeSplit
// or it will represent terminal (leaf) node with defined
// arguments.
//
// Provided arguments are used to determine what is the max
// allowed depth (maxDepth) for constructing tree. Also,
// what is the minimal leaf size (minSize) to end further
// splitting of decision nodes.
//
//
//
func nodeSplit(node *Node, maxDepth, minSize, depth int) {
	left, right := node.Groups[0], node.Groups[1]
	node.Groups = [][][]float64{}

	if len(left) == 0 || len(right) == 0 {
		node.LeftVal = terminalNode(append(left, right...))
		node.RightVal = terminalNode(append(left, right...))
		return
	}

	if depth >= maxDepth {
		node.LeftVal = terminalNode(left)
		node.RightVal = terminalNode(right)
		return
	}

	if len(left) <= minSize {
		node.LeftVal = terminalNode(left)
	} else {
		node.LeftNode = bestSplit(left)
		nodeSplit(node.LeftNode, maxDepth, minSize, depth+1)
	}

	if len(right) <= minSize {
		node.RightVal = terminalNode(right)
	} else {
		node.RightNode = bestSplit(right)
		nodeSplit(node.RightNode, maxDepth, minSize, depth+1)
	}
}

//
// terminalNode determines which class group has statistically
// the greatest likelihood to be the leaf of the current
// decision node.
//
func terminalNode(group [][]float64) float64 {
	classesCount := make(map[float64]int)
	for _, row := range group {
		classesCount[row[len(row)-1]]++
	}

	maxKey := 0.0
	maxVal := 0
	for key, val := range classesCount {
		if val > maxVal {
			maxKey = key
			maxVal = val
		}
	}

	return maxKey
}

//
// giniIndex is used to calculate
// impurity of the each Decision Tree node. It is calculated
// by the following equation:
//
// G = 1 - p(Yes)^2 - p(No)^2
//
// Then, the weighted average of giniIndex impurity for question
// node is calculated as:
//
// [sum of left leaf node] / [sum of both nodes] * [gini impurity of left leaf node] +
// [sum of right leaf node] / [sum of both nodes] * [gini impurity of right leaf node]
//
// giniIndex impurity is calculated for all attribute nodes. Then the
// lowest impurity score is used as a root node of the tree. After that,
// for each node, giniIndex impurity is calculated with existing examples in the
// both left and right directions. This way, decision tree is fully build.
//
func giniIndex(groups [][][]float64, classes []float64) float64 {
	sampleNum := 0
	for _, group := range groups {
		sampleNum += len(group)
	}

	giniScore := 0.0
	for _, group := range groups {
		groupSize := len(group)

		if groupSize == 0 {
			continue
		}

		score := 0.0
		for _, class := range classes {
			classCount := 0
			for _, members := range group {
				// members[len(members)-1] - get class of the current row
				if class == members[len(members)-1] {
					classCount++
				}
			}
			prob := float64(classCount) / float64(groupSize)
			score += math.Pow(prob, 2)
		}
		giniScore += float64(1.0-score) * (float64(groupSize) / float64(sampleNum))
	}

	return giniScore
}

func BaggingPredict(trees []Node, row []float64) float64 {
	predictions := []float64{}
	for _, tree := range trees {
		prediction := tree.Predict(row)
		predictions = append(predictions, prediction)
	}

	maxPrediction := -1.0
	for _, prediction := range predictions {
		if prediction > maxPrediction {
			maxPrediction = prediction
		}
	}

	return maxPrediction
}

//
// Print outputs constructed Decision Tree to the stdout
// with node decisions adn leaf classes.
//
func (node *Node) Print(depth int) {
	fmt.Printf("%s[X%d < %.3f]\n", strings.Repeat(" ", depth), (node.Index + 1), (node.Value))

	if node.LeftNode != nil {
		node.LeftNode.Print(depth + 1)
	}

	if node.RightNode == nil {
		fmt.Printf("%s[%.0f]\n", strings.Repeat(" ", depth+1), node.RightVal)
	}

	if node.LeftNode == nil {
		fmt.Printf("%s[%.0f]\n", strings.Repeat(" ", depth+1), node.LeftVal)
	}

	if node.RightNode != nil {
		node.RightNode.Print(depth + 1)
	}

}

//
// Predict propagate attributes of the dataset row by row through
// the Decision Tree. At the end, when terminal (leaf) node is meet,
// it returns adequate prediction with as terminal node value.
//
func (node *Node) Predict(row []float64) float64 {
	if row[node.Index] < node.Value {
		if node.LeftNode != nil {
			return node.LeftNode.Predict(row)
		} else {
			return node.LeftVal
		}
	} else {
		if node.RightNode != nil {
			return node.RightNode.Predict(row)
		} else {
			return node.RightVal
		}
	}
}

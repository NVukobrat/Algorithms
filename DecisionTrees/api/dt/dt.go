package dt

import (
	"fmt"
	"math"
	"strings"
)

//
// Node structure of the Decision Tree.
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
// BuildTree ...
//
func BuildTree(dataset [][]float64, maxDepth, minSize int) Node {
	rootNode := getSplit(dataset)
	split(rootNode, maxDepth, minSize, 1)

	return *rootNode
}

//
// getSplit algorithm looks for the best possible candidate for split.
// Once best split is found, it can be used as node of the tree.
//
func getSplit(dataset [][]float64) *Node {
	classes := []float64{}
	for _, row := range dataset {
		classes = appendIfMissing(classes, row[len(row)-1])
	}

	bestIndex, bestValue, bestScore, bestGroups := 999, 999.0, 999.0, [][][]float64{}
	for index := 0; index < len(dataset[0])-1; index++ {
		for _, row := range dataset {
			groups := testSplit(index, row[index], dataset)
			gini := giniIndex(groups, classes)
			// fmt.Printf("X%d < %.3f Gini=%.3f\n", index+1, row[index], gini)

			if gini < bestScore {
				bestIndex, bestValue, bestScore, bestGroups = index, row[index], gini, groups
			}
		}
	}

	return &Node{Index: bestIndex, Value: bestValue, Groups: bestGroups}
}

//
// split ...
//
func split(node *Node, maxDepth, minSize, depth int) {
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
		node.LeftNode = getSplit(left)
		split(node.LeftNode, maxDepth, minSize, depth+1)
	}

	if len(right) <= minSize {
		node.RightVal = terminalNode(right)
	} else {
		node.RightNode = getSplit(right)
		split(node.RightNode, maxDepth, minSize, depth+1)
	}
}

//
// terminalNode ...
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

func appendIfMissing(slice []float64, i float64) []float64 {
	for _, ele := range slice {
		if ele == i {
			return slice
		}
	}
	return append(slice, i)
}

//
// testSplit splits dataset in two list of rows giving the index
// and the value of the attribute. Having those two groups, gini
// score could be calculated for given attribute.
//
func testSplit(index int, value float64, dataset [][]float64) [][][]float64 {
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
// Print ...
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
// Predict ...
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

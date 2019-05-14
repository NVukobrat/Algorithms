package main

import (
	"fmt"
	"math"

	"RandomForest/api/rf"
)

func main() {
	// Dataset example.
	dataset := [][]float64{
		[]float64{2.771244718, 1.784783929, 0},
		[]float64{1.728571309, 1.169761413, 0},
		[]float64{3.678319846, 2.81281357, 0},
		[]float64{3.961043357, 2.61995032, 0},
		[]float64{2.999208922, 2.209014212, 0},
		[]float64{7.497545867, 3.162953546, 1},
		[]float64{9.00220326, 3.339047188, 1},
		[]float64{7.4445542326, 0.476683375, 1},
		[]float64{10.12493903, 3.234550982, 1},
		[]float64{6.642287351, 3.319983761, 1},
	}

	// Build Decision Tree.
	featureNum := int(math.Sqrt(float64(len(dataset[0]) - 1)))
	trees := rf.Build(dataset, 9, 1, 10, featureNum)

	// Print constructed Decision Tree.
	tree.Print(0)

	// Give prediction for each row of the dataset.
	for _, row := range dataset {
		prediction := tree.BaggingPredict(trees, row)
		fmt.Printf("Expected=%.0f, Got=%.0f\n", row[len(row)-1], prediction)
	}
}

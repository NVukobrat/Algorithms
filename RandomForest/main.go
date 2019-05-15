package main

import (
	"fmt"
	"math"

	"RandomForest/api/rf"
)

func main() {
	// Dataset example.
	dataset := [][]float64{
		{2.771244718, 1.784783929, 0},
		{1.728571309, 1.169761413, 0},
		{3.678319846, 2.81281357, 0},
		{3.961043357, 2.61995032, 0},
		{2.999208922, 2.209014212, 0},
		{7.497545867, 3.162953546, 1},
		{9.00220326, 3.339047188, 1},
		{7.4445542326, 0.476683375, 1},
		{10.12493903, 3.234550982, 1},
		{6.642287351, 3.319983761, 1},
	}

	// Build Random Forest.
	featureNum := int(math.Sqrt(float64(len(dataset[0]) - 1)))
	trees := rf.Build(dataset, 9, 1, 10, featureNum, 0.8)

	// Print constructed Random Forest Decision Trees.
	fmt.Print("Trees from forest: ")
	for i := 0; i < 10; i++ {
		fmt.Printf("\nTree_%d\n", i+1)
		trees[i].Print(0)
	}

	// Give prediction for each row of the dataset.
	fmt.Println("\nPrediction results: ")
	for _, row := range dataset {
		prediction := rf.BaggingPredict(trees, row)
		fmt.Printf("Expected=%.0f, Got=%.0f\n", row[len(row)-1], prediction)
	}
}

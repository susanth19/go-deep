package deep

import (
	"fmt"
	"math"
)

// GetLoss returns a loss function given a LossType
func GetLoss(loss LossType) Loss {
	switch loss {
	case LossCrossEntropy:
		return CrossEntropy{}
	case LossMeanSquared:
		return MeanSquared{}
	case LossBinaryCrossEntropy:
		return BinaryCrossEntropy{}
	case LossCustom:
		return Custom{}
	}
	return BinaryCrossEntropy{}
}

// LossType represents a loss function
type LossType int

func (l LossType) String() string {
	switch l {
	case LossCrossEntropy:
		return "CE"
	case LossBinaryCrossEntropy:
		return "BinCE"
	case LossMeanSquared:
		return "MSE"
	case LossCustom:
		return "CUSTOM"
	}
	return "N/A"
}

const (
	// LossNone signifies unspecified loss
	LossNone LossType = 0
	// LossCrossEntropy is cross entropy loss
	LossCrossEntropy LossType = 1
	// LossBinaryCrossEntropy is the special case of binary cross entropy loss
	LossBinaryCrossEntropy LossType = 2
	// LossMeanSquared is MSE
	LossMeanSquared LossType = 3
	//LossCustom is CUSTOM
	LossCustom LossType = 4
)

// Loss is satisfied by loss functions
type Loss interface {
	F(estimate, ideal [][]float64) float64
	Df(estimate, ideal, activation float64) float64
}

// CrossEntropy is CE loss
type CrossEntropy struct{}

// F is CE(...)
func (l CrossEntropy) F(estimate, ideal [][]float64) float64 {

	var sum float64
	for i := range estimate {
		ce := 0.0
		for j := range estimate[i] {
			ce += ideal[i][j] * math.Log(estimate[i][j])
		}

		sum -= ce
	}
	fmt.Println(sum)
	return sum / float64(len(estimate))
}

// Df is CE'(...)
func (l CrossEntropy) Df(estimate, ideal, activation float64) float64 {
	return estimate - ideal
}

// BinaryCrossEntropy is binary CE loss
type BinaryCrossEntropy struct{}

// F is CE(...)
func (l BinaryCrossEntropy) F(estimate, ideal [][]float64) float64 {
	epsilon := 1e-16
	var sum float64
	for i := range estimate {
		ce := 0.0
		for j := range estimate[i] {
			ce += ideal[i][j]*math.Log(estimate[i][j]+epsilon) + (1.0-ideal[i][j])*math.Log(1.0-estimate[i][j]+epsilon)
		}
		sum -= ce
	}
	return sum / float64(len(estimate))
}

// Df is CE'(...)
func (l BinaryCrossEntropy) Df(estimate, ideal, activation float64) float64 {
	return estimate - ideal
}

// MeanSquared in MSE loss
type MeanSquared struct{}

// F is MSE(...)
func (l MeanSquared) F(estimate, ideal [][]float64) float64 {
	var sum float64
	fmt.Println(estimate)
	fmt.Println(ideal)
	for i := 0; i < len(estimate); i++ {
		for j := 0; j < len(estimate[i]); j++ {
			sum += math.Pow(estimate[i][j]-ideal[i][j], 2)
		}
	}
	return sum / float64(len(estimate)*len(estimate[0]))
}

// Df is MSE'(...)
func (l MeanSquared) Df(estimate, ideal, activation float64) float64 {
	return activation * (estimate - ideal)
}

//Custom function in CUSTOM loss
type Custom struct{}

//F is CUTSOM'(...)
func (l Custom) F(estimate, ideal [][]float64) float64 {
	var sum float64
	var breach float64 = 2
	var early float64 = 1
	diff := []float64{}
	diff_breach := []float64{}
	diff_early := []float64{}
	diff4 := []float64{}
	for i := 0; i < len(estimate); i++ {

		for j := 0; j < len(estimate[i]); j++ {

			a := ideal[i][j] - estimate[i][j] //actual -predict/
			diff = append(diff, a)
			b := breach * (a * a)
			diff_breach = append(diff_breach, b)
			c := early * (a * a)
			diff_early = append(diff_early, c)
			if a > 0 {
				diff4 = append(diff4, c)
				sum = sum + c
			} else if a < 0 {
				diff4 = append(diff4, b)
				sum = sum + b
			} else {
				diff4 = append(diff4, 0)

			}
		}
	}

	return sum / float64(len(estimate)*len(estimate[0]))

}

//Df is CUSTOM'(...)
func (l Custom) Df(estimate, ideal, activation float64) float64 {
	return activation * (estimate - ideal)
}

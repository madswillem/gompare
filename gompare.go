package gompare

import (
	"fmt"
	"math"
)

func inslice(n string, h []string) bool {
	for _, v := range h {
		if v == n {
			return true
		}
	}
	return false
}

func logical_and(x []string, y []string) []string {
	var log_and []string
	for _, s := range x {
		if inslice(s, y) {
			log_and = append(log_and, s)
		}
	}

	return log_and
}
func logical_or(x []string, y []string) []string {
	var log_or []string
	for _, s := range x {
		if !inslice(s, y) {
			log_or = append(log_or, s)
		}
	}
	for _, s := range y {
		if !inslice(s, x) {
			log_or = append(log_or, s)
		}
	}
	return log_or
}

func JaccardSimilarity(e []string, f []string) float64 {
	observations_in_both := logical_and(e, f)
	observationa_in_either := logical_or(e, f)
	
	return float64(len(observations_in_both)) / float64(len(observationa_in_either))
}

func TfidfVectorizer(d ...[]string) [][]float64 {
	matrix := make([]map[string]float64, len(d))
	for i := range matrix {
		matrix[i] = make(map[string]float64)
	}

	idf_map := make(map[string]float64)

	// Create tf values
	// Setting idf_map to later have a dict of all terms when calculatin idf
	for i := range d {
		for _, s := range d[i] {
			matrix[i][s] += 1.0 / float64(len(d[i]))
			idf_map[s] = 0.0
		}
	}

	// Calculate the number of documents containing the tearm for each term
	for s := range idf_map {
		for i := range d {
			if inslice(s, d[i]) {
				idf_map[s] += 1
			}
		}
	}

	// Calculate
	for i := range d {
		for _, s := range d[i] {
			fmt.Printf("idf value for %s: %f \n", s, math.Log10(float64(len(d)) / idf_map[s]))
			fmt.Printf("tf of %s: %f \n", s, matrix[i][s])
			matrix[i][s] *= math.Log10(float64(len(d)) / 1 + idf_map[s])
		}
	}

	//Build vector
	vector := make([][]float64, len(d))
	for s := range idf_map {
		for i := range matrix {
			vector[i] = append(vector[i], matrix[i][s])
		}
	}

	return vector
}

func CosineSimilarity(v1, v2 []float64) float64 {

	// Calculating A * B
	var dotprodcutAB float64
	for i := range v1 {
		dotprodcutAB += v1[i] * v2[i]
	}

	//Calculating ∥A∥ * ∥B∥
	var magnitudeA float64
	var magnitudeB float64
	var magnitudeAB float64

	for _, f := range v1 {
		magnitudeA += math.Pow(f, 2)
	}
	magnitudeA = math.Sqrt(magnitudeA)

	for _, f := range v2 {
		magnitudeB += math.Pow(f, 2)
	}
	magnitudeB = math.Sqrt(magnitudeB)

	magnitudeAB = magnitudeA * magnitudeB

	return dotprodcutAB / magnitudeAB
}
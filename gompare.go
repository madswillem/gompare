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
			fmt.Printf("idf value for %s: %f \n", s, math.Log10(2))
			matrix[i][s] *= math.Log10(float64(len(d)) / idf_map[s])
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
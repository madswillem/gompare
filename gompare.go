package gompare

import (
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

func createWordMatrix(c [][]string) (map[string]int, [][]float64) {
	dict := make(map[string]int)
	vec := make([][]float64, len(c))

	for _, d := range c {
		for _, s := range d {
			if dict[s] != 0 {
				continue
			}
			dict[s] = len(dict) + 1
		}
	}
	for i := range vec {
		vec[i] = make([]float64, len(dict))
	}

	for i, d := range c {
		for _, s := range d {
			vec[i][dict[s]-1] += 1
		}
	}

	return dict, vec
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
	dict, vec := createWordMatrix(d)
	idf := make(map[string]int, len(dict))
	for i, n := range dict {
		for _, v := range vec {
			idf[i] += int(v[n - 1])
		}
	}

	for i, n := range dict {
		for v := range vec {
			vec[v][n-1] /= float64(len(d[v]))
			vec[v][n-1] *= math.Log10(float64(len(d))/float64(idf[i]))
		}
	}

	return vec
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

func EuclideanDistance(v1, v2 []float64) float64 {
	var ed float64
	for i := range v1 {
		ed += math.Pow(v1[i]-v2[i], 2)
	}

	return math.Sqrt(ed)
}

func OldTfidfVectorizer(d ...[]string) [][]float64 {
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
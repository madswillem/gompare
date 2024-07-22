package gompare

import (
	"math"
	"regexp"
	"strings"
)

type Matrix struct {
	Dict map[string]int
	Vec  [][]float64
}

type Handler struct {
	InputStrings [][]string
	OutputMatrix Matrix
	Similarity   float64
	InputMatrix  Matrix
	Normalizer   func(...string) []string
	Splitter     func(...string) [][]string
}
type Config struct {
	Matrix     Matrix
	Normalizer func(...string) []string
	Splitter   func(...string) [][]string
}

func normalizer(d ...string) []string {
	for i, s := range d {
		// Convert to lowercase
		s = strings.ToLower(s)

		// Remove punctuation
		reg, _ := regexp.Compile(`[^\w\s]`)
		s = reg.ReplaceAllString(s, "")

		// Trim whitespace
		s := strings.TrimSpace(s)

		// Normalize whitespace (convert multiple spaces to a single space)
		d[i] = strings.Join(strings.Fields(s), " ")
	}
	return d
}
func spliter(d ...string) [][]string {
	split := make([][]string, len(d))
	for i, s := range d {
		split[i] = strings.Fields(s)
	}
	return split
}

func New(cfg Config) *Handler {
	h := &Handler{
		InputMatrix: Matrix{},
		Normalizer:  normalizer,
		Splitter:    spliter,
	}
	if cfg.Matrix.Vec != nil {
		h.InputMatrix.Vec = cfg.Matrix.Vec
	}
	if cfg.Matrix.Dict != nil {
		h.InputMatrix.Dict = cfg.Matrix.Dict
	}
	if cfg.Normalizer != nil {
		h.Normalizer = cfg.Normalizer
	}
	if cfg.Splitter != nil {
		h.Splitter = cfg.Splitter
	}

	return h
}

func (h *Handler) Add(d ...string) {
	h.InputStrings = append(h.InputStrings, h.Splitter(h.Normalizer(d...)...)...)
}

func (h *Handler) TfidfMatrix() {
	h.OutputMatrix = TfidfVectorizer(CreateWordMatrix(h.InputStrings, &h.InputMatrix.Dict), h.InputStrings...)
}
func (h *Handler) NormalMatrix() {
	h.OutputMatrix = CreateWordMatrix(h.InputStrings, &h.InputMatrix.Dict)
}
func (h *Handler) CosineSimilarity(d1, d2 int) {
	h.Similarity = CosineSimilarity(h.OutputMatrix.Vec[d1], h.OutputMatrix.Vec[d2])
}
func (h *Handler) EuclideanDistance(d1, d2 int) {
	h.Similarity = EuclideanDistance(h.OutputMatrix.Vec[d1], h.OutputMatrix.Vec[d2])
}

func inslice(n string, h []string) bool {
	for _, v := range h {
		if v == n {
			return true
		}
	}
	return false
}

func CreateWordMatrix(c [][]string, dict *map[string]int) Matrix {
	m := Matrix{}
	m.Vec = make([][]float64, len(c))
	if dict != nil {
		m.Dict = *dict
	}
	m.Dict = make(map[string]int)

	for _, d := range c {
		for _, s := range d {
			if m.Dict[s] != 0 {
				continue
			}
			m.Dict[s] = len(m.Dict) + 1
		}
	}
	for i := range m.Vec {
		m.Vec[i] = make([]float64, len(m.Dict))
	}

	for i, d := range c {
		for _, s := range d {
			m.Vec[i][(m.Dict)[s]-1] += 1
		}
	}

	return m
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

func TfidfVectorizer(m Matrix, d ...[]string) Matrix {

	idf := make(map[string]int, len(m.Dict))
	for i, n := range m.Dict {
		for _, v := range m.Vec {
			idf[i] += int(v[n-1])
		}
	}

	for i, n := range m.Dict {
		for v := range m.Vec {
			m.Vec[v][n-1] /= float64(len(d[v]))
			m.Vec[v][n-1] *= math.Log10(float64(len(d)) / float64(idf[i]))
		}
	}

	return m
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

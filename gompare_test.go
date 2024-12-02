package gompare

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func timer(name string) func() {
	start := time.Now()
	return func() {
		fmt.Printf("%s took %v\n", name, time.Since(start))
	}
}

func TestLogicalAnd(t *testing.T) {
	defer timer("logical_and")()
	a := []string{"hallo", "ich", "bin", "mads"}
	b := []string{"hallo", "tschüss"}
	e := []string{"hallo"}

	r := logical_and(a, b)

	if !reflect.DeepEqual(e, r) {
		t.Fatalf("Result wasnt as expected it was: %s", r)
	}
}
func TestLogicalOR(t *testing.T) {
	defer timer("logical_or")()
	a := []string{"hallo", "ich", "bin", "mads"}
	b := []string{"hallo", "tschüss"}
	e := []string{"ich", "bin", "mads", "tschüss"}

	r := logical_or(a, b)

	if !reflect.DeepEqual(e, r) {
		t.Fatalf("Result wasnt as expected it was: %s", r)
	}
}
func TestJaccardSimalarity(t *testing.T) {
	defer timer("jaccard_simalarity")()
	a := []string{"hallo", "ich", "bin", "mads"}
	b := []string{"hallo", "tschüss"}

	s := JaccardSimilarity(a, b)

	if s != 0.25 {
		t.Fatalf("Result wasnt as expected it was: %f", s)
	}
}

func TestCosineSimilarity(t *testing.T) {
	defer timer("cosine_similarity")()
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}

	r := CosineSimilarity(a, b)

	if r != 0.9746318461970762 {
		t.Fatalf("Expected cosine similarity 0.9746318461970762 return of the cosine similarity function: %f", r)
	}
}

func TestEuclideanDistance(t *testing.T) {
	type args struct {
		v1 []float64
		v2 []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		// TODO: Add test cases.
		{
			name: "test vectors",
			args: args{
				v1: []float64{3, 4},
				v2: []float64{6, 1},
			},
			want: 4.242640687119285,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EuclideanDistance(tt.args.v1, tt.args.v2); got != tt.want {
				t.Errorf("EuclideanDistance() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCreateWordMatrix(t *testing.T) {
	type args struct {
		c [][]string
	}
	tests := []struct {
		name string
		args args
		want Matrix
	}{
		// TODO: Add test cases.
		{
			name: "test CreatWordMatrix",
			args: args{
				c: [][]string{{"hi", "i", "am", "ben"}, {"hi", "bye"}},
			},
			want: Matrix{
				Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
				Vec:  [][]float64{{1, 1, 1, 1, 0}, {1, 0, 0, 0, 1}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := CreateWordMatrix(tt.args.c, nil)
			if !reflect.DeepEqual(m.Dict, tt.want.Dict) {
				t.Errorf("CreateWordMatrix() gotDict = %v, wantDict %v", m.Dict, tt.want.Dict)
			}
			if !reflect.DeepEqual(m.Vec, tt.want.Vec) {
				t.Errorf("CreateWordMatrix() gotVec = %v, wantVec %v", m.Vec, tt.want.Vec)
			}
		})
	}
}

func TestTfidfVectorizer(t *testing.T) {
	type args struct {
		m Matrix
		d [][]string
	}
	tests := []struct {
		name string
		args args
		want Matrix
	}{
		{
			name: "test tfidf vectorizer using matrix",
			args: args{
				m: Matrix{
					Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
					Vec:  [][]float64{{1, 1, 1, 1, 0}, {1, 0, 0, 0, 1}},
				},
				d: [][]string{{"hi", "i", "am", "ben"}, {"hi", "bye"}},
			},
			want: Matrix{
				Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
				Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := TfidfVectorizer(tt.args.m, tt.args.d...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TfidfVectorizer() = %v, want %v. Error: %e", got, tt.want, err)
			}
		})
	}
}

func TestTidfVectorizerWithHandler(t *testing.T) {
	defer timer("tfidf_vectoriter_handler")()
	h := New(Config{})
	want := Matrix{
		Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
		Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
	}

	h.Add("hi i am ben", "hi bye")
	h.TfidfMatrix()

	if !reflect.DeepEqual(h.OutputMatrix, want) {
		t.Errorf("TfidfVectorizer() = %v, want %v", h.OutputMatrix, want)
	}
}

func TestCosineSimilarityHandler(t *testing.T) {
	defer timer("cosine_similarity_handler")()
	h := New(Config{})
	want := 0.0

	h.OutputMatrix = Matrix{
		Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
		Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
	}
	h.CosineSimilarity(0, 1)

	if h.Similarity != want {
		t.Errorf("Cosinesimilarity = %v, want %v", h.Similarity, want)
	}
}
func TestEuclideanDistanceHandler(t *testing.T) {
	defer timer("euclidean_distance_handler")()
	h := New(Config{})
	want := 0.19911262642443656

	h.OutputMatrix = Matrix{
		Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
		Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
	}
	h.EuclideanDistance(0, 1)

	if h.Similarity != want {
		t.Errorf("Euclidean Distance = %v, want %v", h.Similarity, want)
	}
}

func TestHandler_CosineSimilarity(t *testing.T) {
	type fields struct {
		InputStrings [][]string
		OutputMatrix Matrix
		Similarity   float64
		InputMatrix  Matrix
		Normalizer   func(...string) []string
		Splitter     func(...string) [][]string
	}
	type args struct {
		d1 int
		d2 int
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   float64
	}{
		{
			name: "Test CosineSimilarity With Vecs of difrent lengths",
			fields: fields{
				OutputMatrix: Matrix{
					Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
					Vec:  [][]float64{{1, 0, 0, 0, 1}, {1, 1, 1, 1}},
				},
			},
			args: args{
				d1: 0,
				d2: 1,
			},
			want: 0.353553,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{
				InputStrings: tt.fields.InputStrings,
				OutputMatrix: tt.fields.OutputMatrix,
				Similarity:   tt.fields.Similarity,
				InputMatrix:  tt.fields.InputMatrix,
				Normalizer:   tt.fields.Normalizer,
				Splitter:     tt.fields.Splitter,
			}
			h.CosineSimilarity(tt.args.d1, tt.args.d2)

			const tolerance = 1e-6 // Define a small tolerance level

			if math.Abs(tt.want-h.Similarity) > tolerance {
				t.Errorf("Want is %f but h.Similarity is %f", tt.want, h.Similarity)
			}
		})
	}
}

func TestHandler_NormalMatrix(t *testing.T) {
	type fields struct {
		InputStrings [][]string
		OutputMatrix Matrix
		Similarity   float64
		InputMatrix  Matrix
		Normalizer   func(...string) []string
		Splitter     func(...string) [][]string
	}
	tests := []struct {
		name   string
		fields fields
		want   Matrix
	}{
		{
			name: "Test Normal Matrix ",
			fields: fields{
				InputStrings: [][]string{{"hi", "i", "am", "ben"}, {"hi", "bye"}},
			},
			want: Matrix{
				Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
				Vec:  [][]float64{{1, 0, 0, 0, 1}, {1, 1, 1, 1, 0}},
			},
		},
		{
			name: "Test Normal Matrix with input matrix",
			fields: fields{
				InputMatrix: Matrix{
					Dict: map[string]int{"am": 3, "ben": 4, "hi": 1, "i": 2},
					Vec:  [][]float64{{1, 1, 1, 1}},
				},
				InputStrings: [][]string{{"hi", "bye"}},
			},
			want: Matrix{
				Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
				Vec:  [][]float64{{1, 0, 0, 0, 1}, {1, 1, 1, 1}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{
				InputStrings: tt.fields.InputStrings,
				OutputMatrix: tt.fields.OutputMatrix,
				Similarity:   tt.fields.Similarity,
				InputMatrix:  tt.fields.InputMatrix,
				Normalizer:   tt.fields.Normalizer,
				Splitter:     tt.fields.Splitter,
			}
			h.NormalMatrix()

			if !reflect.DeepEqual(h.OutputMatrix.Vec, tt.want.Vec) && !reflect.DeepEqual(h.OutputMatrix.Dict, tt.want.Dict) {
				t.Errorf("%s: want Matrix %v but got %v", tt.name, tt.want, h.OutputMatrix)
			}
		})
	}
}

func TestHandler_TfidfMatrix(t *testing.T) {
	type fields struct {
		InputStrings [][]string
		OutputMatrix Matrix
		Similarity   float64
		InputMatrix  Matrix
		Normalizer   func(...string) []string
		Splitter     func(...string) [][]string
	}
	type want struct {
		M Matrix
		E error
	}
	tests := []struct {
		name   string
		fields fields
		want   want
	}{
		{
			name: "Test Creating TfidfMatrix",
			fields: fields{
				InputStrings: [][]string{{"hi", "i", "am", "ben"}, {"hi", "bye"}},
			},
			want: want{
				M: Matrix{
					Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
					Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
				},
				E: nil,
			},
		},
		{
			name: "Test Creating TfidfMatrix with input matrix",
			fields: fields{
				InputMatrix: Matrix{
					Dict: map[string]int{"am": 3, "ben": 4, "hi": 1, "i": 2},
					Vec:  [][]float64{{1, 1, 1, 1}},
				},
				InputStrings: [][]string{{"hi", "bye"}},
			},
			want: want{
				M: Matrix{
					Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
					Vec:  [][]float64{{1, 0, 0, 0, 1}, {1, 1, 1, 1}},
				},
				E: errors.New("lenght of m and d diffrent"),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{
				InputStrings: tt.fields.InputStrings,
				OutputMatrix: tt.fields.OutputMatrix,
				Similarity:   tt.fields.Similarity,
				InputMatrix:  tt.fields.InputMatrix,
				Normalizer:   tt.fields.Normalizer,
				Splitter:     tt.fields.Splitter,
			}
			err := h.TfidfMatrix()

			if !reflect.DeepEqual(h.OutputMatrix, tt.want.M) {
				t.Errorf("%s: TfidfMatrix() = %v, want %v", tt.name, h.OutputMatrix, tt.want.M)
			}
			if !errors.Is(errors.Unwrap(err), errors.Unwrap(tt.want.E)) {
				t.Errorf("Expected Err %e but got %e", tt.want.E, err)
			}
		})
	}
}

func TestHandler_Add(t *testing.T) {
	input := []string{"Water is the best beverage"}
	expected := [][]string{
		{"water", "best", "beverage"},
	}
	h := New(Config{RemoveDict: Fillerwords_en})

	h.Add(input...)

	if d := cmp.Diff(expected, h.InputStrings); d != "" {
		fmt.Printf("Expected and Actual are different:\n %s", d)
	}
}

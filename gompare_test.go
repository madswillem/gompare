package gompare

import (
	"fmt"
	"reflect"
	"testing"
	"time"
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
	a := []string{"hallo", "ich", "bin", "mads"}
	b := []string{"hallo", "tschüss"}

	s := JaccardSimilarity(a, b)

	if s != 0.25 {
		t.Fatalf("Result wasnt as expected it was: %f", s)
	}
}

func TestCosineSimilarity(t *testing.T) {
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
			if got := TfidfVectorizer(tt.args.m, tt.args.d...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TfidfVectorizer() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTidfVectorizerWithHandler(t *testing.T) {
	h := New(Config{})
	want := Matrix{
		Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
		Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
	}

	h.Add("hi i am ben", "hi bye")
	h.TfidfMatrix()

	if !reflect.DeepEqual(h.OuputMatrix, want) {
		t.Errorf("TfidfVectorizer() = %v, want %v", h.OuputMatrix, want)
	}
}

func TestCosineSimilarityHandler(t *testing.T) {
	h := New(Config{})
	want := 0.0

	h.OuputMatrix = Matrix{
		Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
		Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
	}
	h.CosineSimilarity(0, 1)

	if h.Similarity != want {
		t.Errorf("Cosinesimilarity = %v, want %v", h.Similarity, want)
	}
}
func TestEuclideanDistanceHandler(t *testing.T) {
	h := New(Config{})
	want := 0.19911262642443656

	h.OuputMatrix = Matrix{
		Dict: map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
		Vec:  [][]float64{{0, 0.0752574989159953, 0.0752574989159953, 0.0752574989159953, 0}, {0, 0, 0, 0, 0.1505149978319906}},
	}
	h.EuclideanDistance(0, 1)

	if h.Similarity != want {
		t.Errorf("Euclidean Distance = %v, want %v", h.Similarity, want)
	}
}

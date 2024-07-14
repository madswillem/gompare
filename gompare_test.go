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
func TestTfidfVectorizer(t *testing.T) {
	defer timer("test")()
	a := []string{"hi", "i", "am", "ben"}
	b := []string{"hi", "bye"}

	r := TfidfVectorizer(a, b)

	fmt.Printf("The TFIDf values: %v", r)
}
func TestCosineSimilarity(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}

	r := CosineSimilarity(a, b)

	if r != 0.9746318461970762 {
		t.Fatalf("Expected cosine similarity 0.9746318461970762 return of the cosine similarity function: %f", r)
	}
}
func TestCosineSimilarityWithTFIDF(t *testing.T) {
	a := []string{"hi", "i", "am", "ben"}
	b := []string{"hi", "bye"}

	v := TfidfVectorizer(a, b)
	r := CosineSimilarity(v[0], v[1])

	fmt.Println(r)
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
		name  string
		args  args
		want  map[string]int
		want1 [][]float64
	}{
		// TODO: Add test cases.
		{
			name: "test CreatWordMatrix",
			args: args{
				c: [][]string{{"hi", "i", "am", "ben"}, {"hi", "bye"}},
			},
			want:  map[string]int{"am": 3, "ben": 4, "bye": 5, "hi": 1, "i": 2},
			want1: [][]float64{{1, 1, 1, 1, 0}, {1, 0, 0, 0, 1}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := CreateWordMatrix(tt.args.c)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CreateWordMatrix() got = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("CreateWordMatrix() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

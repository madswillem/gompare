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

func TestTfidfVectorizer(t *testing.T)  {
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
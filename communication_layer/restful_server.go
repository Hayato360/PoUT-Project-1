package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"io/ioutil"
	"bytes"
)

type Task struct {
	ID       string `json:"id"`
	DataBatch string `json:"data_batch"`
	Miner    string `json:"miner"`
}

type TrainingResult struct {
	TaskID string  `json:"task_id"`
	Loss   float64 `json:"loss"`
}

func sendTrainingJob(task Task) (TrainingResult, error) {
	url := "http://localhost:5000/train"
	jsonData, err := json.Marshal(task)
	if err != nil {
		return TrainingResult{}, err
	}

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return TrainingResult{}, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return TrainingResult{}, err
	}

	var result TrainingResult
	if err := json.Unmarshal(body, &result); err != nil {
		return TrainingResult{}, err
	}

	return result, nil
}

func handleTrainingJob(w http.ResponseWriter, r *http.Request) {
	var task Task
	if err := json.NewDecoder(r.Body).Decode(&task); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result, err := sendTrainingJob(task)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(result)
}

func main() {
	http.HandleFunc("/train", handleTrainingJob)

	fmt.Println("RESTful server is running on port 8081")
	http.ListenAndServe(":8081", nil)
}

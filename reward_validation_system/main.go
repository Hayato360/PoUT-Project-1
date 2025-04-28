package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

type TrainingResult struct {
	TaskID string  `json:"task_id"`
	Loss   float64 `json:"loss"`
}

type RewardSystem struct {
	mu          sync.Mutex
	results     map[string]TrainingResult
	threshold   float64
	rewardToken int
}

func NewRewardSystem(threshold float64, rewardToken int) *RewardSystem {
	return &RewardSystem{
		results:     make(map[string]TrainingResult),
		threshold:   threshold,
		rewardToken: rewardToken,
	}
}

func (rs *RewardSystem) ValidateResult(result TrainingResult) bool {
	return result.Loss < rs.threshold
}

func (rs *RewardSystem) IssueReward(result TrainingResult) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	if rs.ValidateResult(result) {
		rs.results[result.TaskID] = result
		fmt.Printf("Reward issued for task %s: %d tokens\n", result.TaskID, rs.rewardToken)
	} else {
		fmt.Printf("Task %s did not meet the threshold. No reward issued.\n", result.TaskID)
	}
}

func (rs *RewardSystem) HandleTrainingResult(w http.ResponseWriter, r *http.Request) {
	var result TrainingResult
	if err := json.NewDecoder(r.Body).Decode(&result); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	rs.IssueReward(result)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "processed"})
}

func main() {
	rs := NewRewardSystem(0.1, 10)

	http.HandleFunc("/training_result", rs.HandleTrainingResult)

	fmt.Println("Reward validation system is running on port 8082")
	http.ListenAndServe(":8082", nil)
}

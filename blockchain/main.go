package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

type Task struct {
	ID       string `json:"id"`
	DataBatch string `json:"data_batch"`
	Miner    string `json:"miner"`
}

type Blockchain struct {
	mu    sync.Mutex
	tasks map[string]Task
}

func NewBlockchain() *Blockchain {
	return &Blockchain{
		tasks: make(map[string]Task),
	}
}

func (bc *Blockchain) CreateTask(dataBatch, miner string) string {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	taskID := fmt.Sprintf("task-%d", len(bc.tasks)+1)
	task := Task{
		ID:       taskID,
		DataBatch: dataBatch,
		Miner:    miner,
	}
	bc.tasks[taskID] = task

	return taskID
}

func (bc *Blockchain) GetTask(taskID string) (Task, bool) {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	task, exists := bc.tasks[taskID]
	return task, exists
}

func (bc *Blockchain) HandleCreateTask(w http.ResponseWriter, r *http.Request) {
	var task Task
	if err := json.NewDecoder(r.Body).Decode(&task); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	taskID := bc.CreateTask(task.DataBatch, task.Miner)
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"task_id": taskID})
}

func (bc *Blockchain) HandleGetTask(w http.ResponseWriter, r *http.Request) {
	taskID := r.URL.Query().Get("task_id")
	if taskID == "" {
		http.Error(w, "task_id is required", http.StatusBadRequest)
		return
	}

	task, exists := bc.GetTask(taskID)
	if !exists {
		http.Error(w, "task not found", http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(task)
}

func main() {
	bc := NewBlockchain()

	http.HandleFunc("/create_task", bc.HandleCreateTask)
	http.HandleFunc("/get_task", bc.HandleGetTask)

	fmt.Println("Blockchain server is running on port 8080")
	http.ListenAndServe(":8080", nil)
}

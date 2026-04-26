import os
import json
from enhanced_main import EnhancedPipeline, CFG

# The Academic Mini-Datasets
DATASETS = {
    "ConflictQA_Simulation": {
        "description": "Testing contradictory claims across different sources (parametric vs contextual).",
        "documents": [
            "A recent 2024 study published in Nature indicates that Vitamin X is highly toxic to the liver in large doses. [year: 2024]",
            "According to the 1990 nutritional guidelines, Vitamin X is completely harmless and beneficial for liver function. [year: 1990]",
            "The 1988 health board review found Vitamin X to be safe and harmless. [year: 1988]"
        ],
        "question": "Is Vitamin X harmful to the liver?"
    },
    "SituatedQA_Simulation": {
        "description": "Testing questions where the answer depends on the temporal snapshot.",
        "documents": [
            "Company XYZ appointed John Doe as CEO in 2015, replacing the founder. [year: 2015]",
            "In 2022, Jane Smith took over as the CEO of Company XYZ after John Doe stepped down. [year: 2022]",
            "Company XYZ was founded by Mark Johnson who served as CEO until 2015. [year: 2010]"
        ],
        "question": "Who was the CEO of Company XYZ in 2018?"
    }
}

def run_benchmark_tests():
    print("\n" + "="*60)
    print("🎓 TRUTHFULRAG V5 - ACADEMIC BENCHMARK SIMULATION")
    print("="*60)
    
    # Initialize the v5 pipeline
    print("Initializing EnhancedPipeline...")
    pipeline = EnhancedPipeline(CFG)
    
    for benchmark_name, data in DATASETS.items():
        print(f"\n🚀 Running Benchmark: {benchmark_name}")
        print(f"Description: {data['description']}")
        print(f"Question: {data['question']}\n")
        
        # 1. Ingest
        print("Ingesting dataset documents into Neo4j...")
        pipeline.build(data["documents"])
        
        # 2. Retrieve & Resolve
        print("Retrieving and resolving conflicts...")
        result = pipeline.ask(data["question"])
        
        print("-" * 40)
        print("v5 FINAL OUTPUT:")
        print(f"Answer: {result.get('answer', 'N/A').strip()}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        if result.get("chain"):
            print(f"\nExplanation Chain:\n{result['chain']}")
        print("-" * 40)

if __name__ == "__main__":
    run_benchmark_tests()

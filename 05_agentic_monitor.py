import json
from agentic_core import run_root_cause_agent, run_routing_agent

if __name__ == "__main__":
    print("=== ROOT-CAUSE INVESTIGATION AGENT ===\n")
    for report in run_root_cause_agent():
        print(json.dumps(report, indent=2))
        print()

    print("=== ALTERNATIVE-ROUTING RECOMMENDATION AGENT ===\n")
    for report in run_routing_agent():
        print(json.dumps(report, indent=2))
        print()

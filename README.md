# 🤖 Multi-Agent Inventory System

An AI-powered inventory management system where multiple autonomous agents collaborate to handle stock monitoring, pricing, and order management — built for **Munder Difflin** paper supplies.

---

## 🧠 How It Works

Instead of one monolithic service, this system uses **specialized AI agents**, each with a single responsibility:

| Agent | Role |
|---|---|
| `inventory_agent` | Checks stock levels and availability |
| `quoting_agent` | Calculates pricing, markups, and discounts |

Agents are powered by **GPT-4o-mini** via the OpenAI API and orchestrated using **smolagents**.

---

## ⚙️ Tech Stack

- **Python 3.11+**
- **smolagents** — multi-agent orchestration
- **OpenAI API** — GPT-4o-mini for agent reasoning
- **SQLAlchemy** — database ORM
- **SQLite** — local inventory database
- **Pandas / NumPy** — data processing
- **python-dotenv** — environment variable management

---

## 📁 Project Structure

```
multi-agent-inventory-system/
├── app.py               # Main entry point
├── requirements.txt     # Project dependencies
├── .env                 # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/himanshu-mishu/multi-agent-inventory-system.git
cd multi-agent-inventory-system
```

### 2. Create and activate virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install smolagents
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the app
```bash
python3 app.py
```

---

## 📦 Example Output

```
Request: Check inventory for Glossy paper and generate a quote for 200 units

inventory_agent  →  Current Stock: 587 units | Status: Available
quoting_agent    →  Unit Price: $0.20 | Final Quote: $54.00 | Delivery: 3-5 days
```

---

## 💡 Key Concepts

- **Multi-agent architecture** — decoupled agents, each does one job
- **Autonomous decision-making** — agents reason and act without human input
- **Tool use** — agents call custom tools (`check_stock_tool`, `calculate_quote_tool`)
- **Real-time inventory** — stock levels fetched live from SQLite database

---

## 📋 Requirements

- Python 3.10+
- OpenAI API key
- Mac / Linux / Windows

---

## 👨‍💻 Author

**Himanshu Kumar**  
[github.com/himanshu-mishu](https://github.com/himanshu-mishu)

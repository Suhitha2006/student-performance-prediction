import { useState } from 'react'
import './App.css'

function App() {
  const [formData, setFormData] = useState({
    gender: 'female',
    race_ethnicity: 'group C',
    parental_education: 'some college',
    lunch: 'standard',
    test_preparation_course: 'none'
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handlePredict = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const data = await response.json()
      if (response.ok) {
        setResult(data)
      } else {
        alert(data.detail || "Error connecting to backend")
      }
    } catch (error) {
      alert("Failed to connect to the prediction API. Is the FastAPI backend running?")
    }
    setLoading(false)
  }

  return (
    <div className="dashboard">
      <header className="header">
        <div className="badge">ML CLASSIFICATION PROJECT</div>
        <h1>Student Performance <span>Predictor</span></h1>
        <p>Predicting Student Outcomes with Machine Learning</p>
      </header>

      <main className="main-content">
        <div className="left-panel">
          <div className="input-card">
            <h3>Predict Outcome</h3>
            <p>Enter Student Details</p>
            
            <div className="form-group">
              <label>Gender</label>
              <select name="gender" value={formData.gender} onChange={handleChange}>
                <option value="female">Female</option>
                <option value="male">Male</option>
              </select>
            </div>

            <div className="form-group">
              <label>Race / Ethnicity</label>
              <select name="race_ethnicity" value={formData.race_ethnicity} onChange={handleChange}>
                <option value="group A">Group A</option>
                <option value="group B">Group B</option>
                <option value="group C">Group C</option>
                <option value="group D">Group D</option>
                <option value="group E">Group E</option>
              </select>
            </div>

            <div className="form-group">
              <label>Parental Education</label>
              <select name="parental_education" value={formData.parental_education} onChange={handleChange}>
                <option value="some high school">Some High School</option>
                <option value="high school">High School</option>
                <option value="some college">Some College</option>
                <option value="associate's degree">Associate's Degree</option>
                <option value="bachelor's degree">Bachelor's Degree</option>
                <option value="master's degree">Master's Degree</option>
              </select>
            </div>

            <div className="form-group">
              <label>Lunch Type</label>
              <select name="lunch" value={formData.lunch} onChange={handleChange}>
                <option value="standard">Standard</option>
                <option value="free/reduced">Free/Reduced</option>
              </select>
            </div>

            <div className="form-group">
              <label>Test Prep Course</label>
              <select name="test_preparation_course" value={formData.test_preparation_course} onChange={handleChange}>
                <option value="none">None</option>
                <option value="completed">Completed</option>
              </select>
            </div>

            <button className="predict-btn" onClick={handlePredict} disabled={loading}>
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>
        </div>

        <div className="right-panel">
          <div className="result-card">
            <h3>Prediction Result</h3>
            {result ? (
              <div className="score-breakdown">
                <div className={`status ${result.overall_status.toLowerCase()}`}>
                  {result.overall_status}
                </div>
                <div className="metrics-grid">
                  <div className="metric">
                    <span className="label">Math Score</span>
                    <span className="value">{result.math_score}</span>
                  </div>
                  <div className="metric">
                    <span className="label">Reading Score</span>
                    <span className="value">{result.reading_score}</span>
                  </div>
                  <div className="metric">
                    <span className="label">Writing Score</span>
                    <span className="value">{result.writing_score}</span>
                  </div>
                  <div className="metric">
                    <span className="label">Average</span>
                    <span className="value">{result.average_score}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                Enter details and click Predict to see the estimated academic outcomes.
              </div>
            )}
          </div>
          
          <div className="api-card">
            <div className="api-header">
              <span className="icon">⚡</span>
              <h3>FastAPI</h3>
            </div>
            <p>Instant Predictions API</p>
            <div className="code-block">
              POST /predict
              <br/>
              <pre>{JSON.stringify(result || {"prediction": "Pass", "math": 85.0}, null, 2)}</pre>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

// Global variables
let clickRatesChart = null;
let deviceChart = null;
let trendsChart = null;

// DOM elements
const predictionForm = document.getElementById('predictionForm');
const predictionResult = document.getElementById('predictionResult');
const trainModelBtn = document.getElementById('trainModelBtn');
const loadAnalyticsBtn = document.getElementById('loadAnalyticsBtn');
const modelTrainingResult = document.getElementById('modelTrainingResult');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    predictionForm.addEventListener('submit', handlePrediction);
    trainModelBtn.addEventListener('click', handleTrainModel);
    loadAnalyticsBtn.addEventListener('click', handleLoadAnalytics);
    
    // Load analytics on page load
    handleLoadAnalytics();
});

// Handle prediction form submission
async function handlePrediction(e) {
    e.preventDefault();
    
    const formData = new FormData(predictionForm);
    const userData = {
        age: parseInt(formData.get('age')),
        daily_time_spent: parseFloat(formData.get('daily_time_spent')),
        area_income: parseFloat(formData.get('area_income')),
        daily_internet_usage: parseFloat(formData.get('daily_internet_usage')),
        gender: formData.get('gender'),
        device_type: formData.get('device_type'),
        ad_topic: formData.get('ad_topic')
    };
    
    // Validate form data
    if (!userData.gender || !userData.device_type || !userData.ad_topic) {
        showError('Please fill in all fields');
        return;
    }
    
    // Show loading state
    const submitBtn = predictionForm.querySelector('button[type="submit"]');
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictionResult(data.prediction);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        // Reset loading state
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
}

// Display prediction results
function displayPredictionResult(prediction) {
    const probabilityValue = document.getElementById('probabilityValue');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceValue = document.getElementById('confidenceValue');
    
    const probability = Math.round(prediction.probability * 100);
    const label = prediction.prediction === 1 ? 'Will Click' : 'Will Not Click';
    const confidence = Math.round(prediction.confidence * 100);
    
    probabilityValue.textContent = probability + '%';
    predictionLabel.textContent = label;
    confidenceValue.textContent = confidence + '%';
    
    // Update probability circle color based on probability
    const probabilityCircle = document.querySelector('.probability-circle');
    if (probability > 70) {
        probabilityCircle.style.background = 'conic-gradient(from 0deg, #48bb78 0deg, #38a169 360deg)';
    } else if (probability > 40) {
        probabilityCircle.style.background = 'conic-gradient(from 0deg, #ed8936 0deg, #dd6b20 360deg)';
    } else {
        probabilityCircle.style.background = 'conic-gradient(from 0deg, #f56565 0deg, #e53e3e 360deg)';
    }
    
    predictionResult.style.display = 'block';
    predictionResult.scrollIntoView({ behavior: 'smooth' });
}

// Handle model training
async function handleTrainModel() {
    trainModelBtn.classList.add('loading');
    trainModelBtn.disabled = true;
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayTrainingResults(data.results);
            showSuccess('Model trained successfully!');
        } else {
            showError(data.error || 'Training failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        trainModelBtn.classList.remove('loading');
        trainModelBtn.disabled = false;
    }
}

// Display training results
function displayTrainingResults(results) {
    const modelAccuracy = document.getElementById('modelAccuracy');
    const modelPrecision = document.getElementById('modelPrecision');
    const modelRecall = document.getElementById('modelRecall');
    
    modelAccuracy.textContent = (results.accuracy * 100).toFixed(1) + '%';
    
    // Get precision and recall from classification report
    const classReport = results.classification_report;
    if (classReport && classReport['1']) {
        modelPrecision.textContent = (classReport['1']['precision'] * 100).toFixed(1) + '%';
        modelRecall.textContent = (classReport['1']['recall'] * 100).toFixed(1) + '%';
    } else {
        modelPrecision.textContent = 'N/A';
        modelRecall.textContent = 'N/A';
    }
    
    modelTrainingResult.style.display = 'block';
    modelTrainingResult.scrollIntoView({ behavior: 'smooth' });
}

// Handle analytics loading
async function handleLoadAnalytics() {
    loadAnalyticsBtn.classList.add('loading');
    loadAnalyticsBtn.disabled = true;
    
    try {
        const response = await fetch('/analytics');
        const data = await response.json();
        
        if (data.success) {
            displayAnalytics(data.analytics);
        } else {
            showError(data.error || 'Failed to load analytics');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        loadAnalyticsBtn.classList.remove('loading');
        loadAnalyticsBtn.disabled = false;
    }
}

// Display analytics charts
function displayAnalytics(analytics) {
    createClickRatesChart(analytics.click_rates_by_age);
    createDeviceChart(analytics.click_rates_by_device);
    createTrendsChart(analytics.daily_trends);
}

// Create click rates by age chart
function createClickRatesChart(data) {
    const ctx = document.getElementById('clickRatesChart').getContext('2d');
    
    if (clickRatesChart) {
        clickRatesChart.destroy();
    }
    
    clickRatesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: 'Click Rate',
                data: Object.values(data),
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(118, 75, 162, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)'
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(118, 75, 162, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Click Rates by Age Group'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Create device chart
function createDeviceChart(data) {
    const ctx = document.getElementById('deviceChart').getContext('2d');
    
    if (deviceChart) {
        deviceChart.destroy();
    }
    
    deviceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(118, 75, 162, 0.8)',
                    'rgba(255, 99, 132, 0.8)'
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(118, 75, 162, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Click Rates by Device Type'
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Create trends chart
function createTrendsChart(data) {
    const ctx = document.getElementById('trendsChart').getContext('2d');
    
    if (trendsChart) {
        trendsChart.destroy();
    }
    
    trendsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(item => item.day),
            datasets: [{
                label: 'Daily Clicks',
                data: data.map(item => item.clicks),
                borderColor: 'rgba(102, 126, 234, 1)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Daily Click Trends'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Utility functions
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    
    // Remove existing error messages
    const existingErrors = document.querySelectorAll('.error');
    existingErrors.forEach(error => error.remove());
    
    // Add new error message
    const container = document.querySelector('.container');
    container.insertBefore(errorDiv, container.firstChild);
    
    // Remove error after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success';
    successDiv.textContent = message;
    
    // Remove existing success messages
    const existingSuccess = document.querySelectorAll('.success');
    existingSuccess.forEach(success => success.remove());
    
    // Add new success message
    const container = document.querySelector('.container');
    container.insertBefore(successDiv, container.firstChild);
    
    // Remove success after 3 seconds
    setTimeout(() => {
        successDiv.remove();
    }, 3000);
}
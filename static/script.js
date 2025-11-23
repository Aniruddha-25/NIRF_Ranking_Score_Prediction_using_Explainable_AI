// Load institutes on page load
window.onload = function() {
    fetch('/get_institutes')
        .then(r => r.json())
        .then(data => {
            const select = document.getElementById('institute');
            data.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                select.appendChild(opt);
            });
        });
};

function predict() {
    const institute = document.getElementById('institute').value;
    if (!institute) {
        document.getElementById('error-message').innerHTML = '<div class="error">Please select an institute</div>';
        return;
    }
    
    document.getElementById('error-message').innerHTML = '';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').classList.remove('show');

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ institute_name: institute })
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        if (data.error) {
            document.getElementById('error-message').innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }

        // Institute Info
        let infoHtml = '<tr><th>Field</th><th>Value</th></tr>';
        Object.entries(data.institute_info).forEach(([k, v]) => {
            infoHtml += `<tr><td>${k}</td><td>${v}</td></tr>`;
        });
        document.getElementById('institute-info').innerHTML = infoHtml;

        // Present Year
        let presentHtml = '<tr><th>Year</th><th>Score</th><th>Rank</th><th>Movement</th></tr>';
        presentHtml += `<tr><td>${data.present.year}</td><td>${data.present.score}</td><td>${data.present.rank}</td><td>${data.present.movement}</td></tr>`;
        document.getElementById('present-year').innerHTML = presentHtml;

        // Forecast
        let forecastHtml = '<tr><th>Year</th><th>Score</th><th>Rank</th><th>Movement</th></tr>';
        data.forecast.forEach(row => {
            forecastHtml += `<tr><td>${row.year}</td><td>${row.score}</td><td>${row.rank}</td><td>${row.movement}</td></tr>`;
        });
        document.getElementById('forecast').innerHTML = forecastHtml;

        // Conclusion
        const conclusionDiv = document.getElementById('conclusion');
        conclusionDiv.textContent = data.conclusion;
        conclusionDiv.className = 'conclusion';
        if (data.conclusion.includes('DECLINE')) conclusionDiv.className += ' decline';
        else if (data.conclusion.includes('STABLE')) conclusionDiv.className += ' stable';

        document.getElementById('result').classList.add('show');
    })
    .catch(err => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('error-message').innerHTML = `<div class="error">Error: ${err.message}</div>`;
    });
}

document.addEventListener('DOMContentLoaded', () => {
    updateObjectCount();

    // Example function to simulate showing the labeling form
    // Replace this with actual logic to show the form when unknown objects are detected
    function simulateUnknownObject() {
        const simulatedObjectId = 1; // Example ID
        showLabelingForm(simulatedObjectId);
    }

    // Simulate showing the labeling form after 5 seconds for testing
    setTimeout(simulateUnknownObject, 5000);

    // Event listener for form submission
    document.getElementById('label-form').addEventListener('submit', handleLabelSubmission);
});

function updateObjectCount() {
    fetch('/object_count')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('object-count-list');
            list.innerHTML = '';
            for (const [label, count] of Object.entries(data)) {
                const listItem = document.createElement('li');
                listItem.textContent = `${label}: ${count} times`;
                list.appendChild(listItem);
            }
        })
        .catch(error => console.error('Error fetching object count:', error));
}

function handleLabelSubmission(event) {
    event.preventDefault();
    const objectId = document.getElementById('object-id').value;
    const label = document.getElementById('object-label').value;

    fetch('/label_object', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ object_id: objectId, label: label })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById('labeling-container').style.display = 'none';
            updateObjectCount();
        }
    })
    .catch(error => console.error('Error labeling object:', error));
}

function showLabelingForm(objectId) {
    document.getElementById('object-id').value = objectId;
    document.getElementById('labeling-container').style.display = 'block';
}

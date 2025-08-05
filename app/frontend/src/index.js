const API_BASE_URL = 'http://localhost:5000/api'; // Flask backend URL

export async function getHistoricalPrices() {
  const response = await fetch(`${API_BASE_URL}/historical-prices`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
}

export async function getChangePointData() {
  const response = await fetch(`${API_BASE_URL}/change-point`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
}

export async function getEventsData() {
  const response = await fetch(`${API_BASE_URL}/events`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
}

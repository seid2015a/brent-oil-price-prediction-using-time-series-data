import React from 'react';
import './EventTable.css'; // For basic styling

const EventTable = ({ events, changePointDate }) => {
  // Sort events by date for better presentation
  const sortedEvents = [...events].sort((a, b) => new Date(a.Date) - new Date(b.Date));

  return (
    <div className="event-table-container">
      <h3>Events List</h3>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Event Type</th>
            <th>Description</th>
            <th>Relevance to Change Point</th>
          </tr>
        </thead>
        <tbody>
          {sortedEvents.map((event, index) => {
            const isNearChangePoint = changePointDate &&
              Math.abs(new Date(event.Date).getTime() - new Date(changePointDate).getTime()) < (60 * 24 * 60 * 60 * 1000); // 60 days in milliseconds
            return (
              <tr key={index} className={isNearChangePoint ? 'highlight-event' : ''}>
                <td>{event.Date}</td>
                <td>{event.EventType}</td>
                <td>{event.EventDescription}</td>
                <td>{isNearChangePoint ? 'Yes (within 60 days)' : 'No'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default EventTable;

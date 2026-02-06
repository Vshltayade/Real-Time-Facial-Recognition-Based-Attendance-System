fetch("http://127.0.0.1:5000/attendance")
  .then(res => res.json())
  .then(data => {
    const table = document.getElementById("records");
    table.innerHTML = "";

    if (data.length === 0) {
      table.innerHTML =
        "<tr><td colspan='4'>No attendance records</td></tr>";
      return;
    }

    data.forEach(row => {
      table.innerHTML += `
        <tr>
          <td>${row.student_name}</td>
          <td>${row.date}</td>
          <td>${row.time}</td>
          <td>${row.course_name || '-'}</td>
        </tr>
      `;
    });
  })
  .catch(err => console.error("Error:", err));

  function exportCSV() {
  fetch("http://127.0.0.1:5000/attendance")
    .then(res => res.json())
    .then(data => {
      if (data.length === 0) {
        alert("No data to export");
        return;
      }

      let csv = "Name,Date,Time,Course\n";

      data.forEach(row => {
        csv += `${row.student_name},${row.date},${row.time},${row.course_name}\n`;
      });

      const blob = new Blob([csv], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "attendance_records.csv";
      a.click();

      window.URL.revokeObjectURL(url);
    })
    .catch(err => console.error("CSV Export Error:", err));
}

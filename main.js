function calculateDays() {
  var date1 = new Date(document.getElementById("date1").value);
  var date2 = new Date(document.getElementById("date2").value);
  var diff = Math.abs(date1.getTime() - date2.getTime());
  var days = Math.ceil(diff / (1000 * 3600 * 24));
  document.getElementById("result").innerHTML = days;
}
/* The sign-in form of the deny page (served without a session, with its own
 * strict CSP): the token goes to POST login, which answers with a session
 * cookie; the page then loads MACE GUI. Nothing else runs here. */
(function () {
  "use strict";
  var form = document.getElementById("signin");
  if (!form) return;
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var err = document.getElementById("err");
    var tok = document.getElementById("token").value.trim();
    if (!tok) { err.textContent = "Paste the token first."; return; }
    fetch("login", { method: "POST", credentials: "same-origin",
      headers: { "X-Requested-With": "mace-gui", "Content-Type": "application/json" },
      body: JSON.stringify({ token: tok }) })
      .then(function (r) {
        if (r.ok) { location.replace("./"); return; }
        err.textContent = r.status === 429 ? "Too many attempts at once; wait a moment." : "That token is not right.";
      })
      .catch(function () { err.textContent = "The server did not answer."; });
  });
})();

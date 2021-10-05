function tabswitch(tab) {
    if (tab !== "board" && tab !== "members")
        return;

    if (tab === "board") {
        document.getElementById("tab-members").removeAttribute("onfocus");
        document.getElementById("tab-board").setAttribute("onfocus", "");
        document.getElementById("tab-content-members").removeAttribute("onfocus");
        document.getElementById("tab-content-board").setAttribute("onfocus", "");
    } else if (tab === "members") {
        document.getElementById("tab-board").removeAttribute("onfocus");
        document.getElementById("tab-members").setAttribute("onfocus", "");
        document.getElementById("tab-content-board").removeAttribute("onfocus");
        document.getElementById("tab-content-members").setAttribute("onfocus", "");
    }
}
async function uploadPDF() {

    const fileInput =
        document.getElementById("pdfFile")

    const formData = new FormData()

    formData.append(
        "file",
        fileInput.files[0]
    )

    await fetch("/upload", {
        method: "POST",
        body: formData
    })

    alert("Uploaded")
}

async function askQuestion() {

    const question =
        document.getElementById("question").value

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            question: question
        })
    })

    const data = await response.json()

    document.getElementById("response")
        .innerText = data.response
}
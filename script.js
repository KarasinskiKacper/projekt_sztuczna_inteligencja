const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  const reader = new FileReader();

  reader.onload = () => {
    preview.src = reader.result;
  };

  reader.readAsDataURL(file);
});

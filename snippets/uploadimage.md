# Upload een afbeelding en toon deze in de DOM

Maak een image input field. Op mobiel kan je de camera als input gebruiken!

```html
<div>
  <label for="file">Upload Image</label>
  <input type="file" accept="image/*;capture=camera" id="file">
</div>

<div><img id="output" width="400"/></div>
```

Als iemand op de file input klikt, kan je de `src` van de image veranderen in het geselecteerde bestand.

```javascript
const image = document.getElementById('output')
const fileButton = document.querySelector("#file")

fileButton.addEventListener("change", (event)=>{
    image.src = URL.createObjectURL(event.target.files[0])
})
```
⚠️ Let op! Je kan de image pas gebruiken nadat het volledig is ingeladen. Dat kan je checken met het `load` event:

```javascript
image.addEventListener('load', () => userImageUploaded())

function userImageUploaded(){
    console.log("The image is now visible in the DOM")
}
```



> 🤯  Tip! Je kan meteen testen op je mobiel door je lokale IP adres te typen in plaats van localhost. Bv: `http://192.168.2.4/prg8/test/`. Je hoeft je project dan niet telkens te uploaden naar een live server.

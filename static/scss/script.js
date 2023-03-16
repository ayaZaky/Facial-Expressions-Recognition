
function getFileName()
{
    var x = document.getElementById('entry_value')
    document.getElementById('fileName').innerHTML = x.value.split('\\').pop()
}

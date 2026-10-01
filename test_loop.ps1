# This script fulfills the strict rule: "verify stability by running main code at least 5 times in a loop"
for ($i=1; $i -le 5; $i++) {
    Write-Host "`n=== Test Run $i ==="
    # Provide predefined inputs mapped to the questions prompted by predict_cli.py
    # Brand, Year, Fuel, KMs, Transmission
    $inputs = "Hyundai`n2018`nDiesel`n60000`nAutomatic`n"
    $inputs | python predict_cli.py
}
Write-Host "`n=== Loop Testing Completed ==="

# frozen_string_literal: true

require 'ipaddr'

load '/workspace/app/lib/private_address_check.rb'

address = IPAddr.new('::')
blocked = PrivateAddressCheck.private_address?(address)

unless blocked
  warn 'IPv6 unspecified address :: was accepted as non-private'
  exit 1
end

puts 'IPv6 unspecified address :: was blocked'
